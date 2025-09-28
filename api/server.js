import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import fetch from 'node-fetch';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import Joi from 'joi';

dotenv.config();

const app = express();


app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:8080',
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const checkInteractionsSchema = Joi.object({
  medications: Joi.array().items(Joi.string()).min(2).required(),
  patientId: Joi.string().uuid().optional()
});

// Health check endpoint
app.get('/api/health', async (req, res) => {
  let mlServiceHealthy = false;
  let mlServiceDetails = {
    sentence_transformers: false,
    openai: false
  };

  // Test ML service connection and get detailed status
  try {
    const mlServiceUrl = process.env.ML_BASE || 'http://localhost:8000';
    const mlResponse = await fetch(`${mlServiceUrl}/health`, { timeout: 2000 });

    if (mlResponse.ok) {
      mlServiceHealthy = true;
      const mlData = await mlResponse.json();

      // Extract detailed service status from ML service
      if (mlData.services) {
        mlServiceDetails.sentence_transformers = mlData.services.sentence_transformers || false;
        mlServiceDetails.openai = mlData.services.openai || false;
      }
    }
  } catch (error) {
    console.log('ML service health check failed:', error.message);
    mlServiceHealthy = false;
  }

  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {
      supabase: !!process.env.SUPABASE_URL,
      ml_service: mlServiceHealthy,
      sentence_transformers: mlServiceDetails.sentence_transformers,
      openai: mlServiceDetails.openai
    }
  });
});

// Get all patients
app.get('/api/patients', async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('patients')
      .select('*')
      .order('name');

    if (error) throw error;
    res.json({ patients: data });
  } catch (error) {
    console.error('Error fetching patients:', error);
    res.status(500).json({ error: 'Failed to fetch patients' });
  }
});

// Get specific patient
app.get('/api/patients/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const { data, error } = await supabase
      .from('patients')
      .select('*')
      .eq('id', id)
      .single();

    if (error) throw error;
    if (!data) return res.status(404).json({ error: 'Patient not found' });

    res.json({ patient: data });
  } catch (error) {
    console.error('Error fetching patient:', error);
    res.status(500).json({ error: 'Failed to fetch patient' });
  }
});

// Get drug information
app.get('/api/drugs/:name', async (req, res) => {
  try {
    const { name } = req.params;

    const { data, error } = await supabase
      .from('drugs')
      .select('*')
      .ilike('name', name)
      .single();

    if (error) throw error;
    if (!data) return res.status(404).json({ error: 'Drug not found' });

    res.json({ drug: data });
  } catch (error) {
    console.error('Error fetching drug:', error);
    res.status(500).json({ error: 'Failed to fetch drug information' });
  }
});

// Get all drugs
app.get('/api/drugs', async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('drugs')
      .select('name, common_uses')
      .order('name');

    if (error) throw error;
    res.json({ drugs: data });
  } catch (error) {
    console.error('Error fetching drugs:', error);
    res.status(500).json({ error: 'Failed to fetch drugs' });
  }
});

// Search drugs using RxNorm API
app.get('/api/drugs/search/:query', async (req, res) => {
  try {
    const { query } = req.params;

    // Call RxNorm API to search for drugs
    const rxnormUrl = `https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=${encodeURIComponent(query)}&maxEntries=10`;
    const rxnormResponse = await fetch(rxnormUrl);

    if (!rxnormResponse.ok) {
      throw new Error('RxNorm API request failed');
    }

    const rxnormData = await rxnormResponse.json();
    const suggestions = rxnormData.approximateGroup?.candidate || [];

    res.json({
      suggestions: suggestions.map(drug => ({
        rxcui: drug.rxcui,
        name: drug.name,
        score: drug.score
      }))
    });
  } catch (error) {
    console.error('Error searching drugs:', error);
    res.status(500).json({ error: 'Failed to search drugs' });
  }
});

// Get detailed drug information from RxNorm
app.get('/api/drugs/details/:rxcui', async (req, res) => {
  try {
    const { rxcui } = req.params;

    // Get drug properties from RxNorm
    const propertiesUrl = `https://rxnav.nlm.nih.gov/REST/rxcui/${rxcui}/properties.json`;
    const propertiesResponse = await fetch(propertiesUrl);

    let drugInfo = { rxcui, name: 'Unknown Drug' };

    if (propertiesResponse.ok) {
      const propertiesData = await propertiesResponse.json();
      if (propertiesData.properties) {
        drugInfo = {
          rxcui: propertiesData.properties.rxcui,
          name: propertiesData.properties.name,
          synonym: propertiesData.properties.synonym,
          tty: propertiesData.properties.tty
        };
      }
    }

    // Try to get adverse events from OpenFDA (optional)
    let adverseEvents = [];
    try {
      const fdaUrl = `https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:"${encodeURIComponent(drugInfo.name)}"&limit=10`;
      const fdaResponse = await fetch(fdaUrl);

      if (fdaResponse.ok) {
        const fdaData = await fdaResponse.json();
        if (fdaData.results) {
          adverseEvents = fdaData.results.slice(0, 5).map(event => ({
            reactions: event.patient?.reaction?.map(r => r.reactionmeddrapt) || [],
            reportdate: event.reportdate
          }));
        }
      }
    } catch (fdaError) {
      console.log('OpenFDA request failed, continuing without adverse events data:', fdaError.message);
    }

    res.json({
      drug: drugInfo,
      adverseEvents: adverseEvents
    });
  } catch (error) {
    console.error('Error getting drug details:', error);
    res.status(500).json({ error: 'Failed to get drug details' });
  }
});

// Add drug to patient
app.post('/api/patients/:id/drugs', async (req, res) => {
  try {
    const { id: patientId } = req.params;
    const { drugName, rxcui } = req.body;

    if (!drugName) {
      return res.status(400).json({ error: 'Drug name is required' });
    }

    // Get current patient data
    const { data: patient, error: patientError } = await supabase
      .from('patients')
      .select('*')
      .eq('id', patientId)
      .single();

    if (patientError || !patient) {
      return res.status(404).json({ error: 'Patient not found' });
    }

    // Add new drug to medications array
    const updatedMedications = [...(patient.medications || []), drugName];

    // Update patient in database
    const { data: updatedPatient, error: updateError } = await supabase
      .from('patients')
      .update({
        medications: updatedMedications,
        updated_at: new Date().toISOString()
      })
      .eq('id', patientId)
      .select()
      .single();

    if (updateError) {
      throw updateError;
    }

    // Check for interactions with new drug
    const interactionCheckUrl = process.env.ML_BASE || 'http://localhost:8000';
    let interactionResults = [];

    // Check new drug against each existing medication
    for (const existingMed of patient.medications || []) {
      try {
        const mlResponse = await fetch(`${interactionCheckUrl}/interactions/check-enhanced`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            drugA: existingMed,
            drugB: drugName,
            patientContext: {
              age: patient.age,
              conditions: patient.conditions || [],
              allMedications: updatedMedications
            }
          }),
          timeout: 10000
        });

        if (mlResponse.ok) {
          const mlResult = await mlResponse.json();
          interactionResults.push({
            existingDrug: existingMed,
            newDrug: drugName,
            ...mlResult
          });
        }
      } catch (mlError) {
        console.log(`ML service error for ${existingMed} + ${drugName}:`, mlError.message);
        // Fallback to basic analysis
        interactionResults.push({
          existingDrug: existingMed,
          newDrug: drugName,
          severity: 'unknown',
          description: `Interaction check unavailable for ${existingMed} + ${drugName}`,
          recommendation: 'Manual review recommended',
          method: 'fallback'
        });
      }
    }

    res.json({
      patient: updatedPatient,
      interactionResults: interactionResults,
      summary: {
        newDrug: drugName,
        addedSuccessfully: true,
        interactionsFound: interactionResults.filter(r => r.severity !== 'mild').length,
        totalChecks: interactionResults.length
      }
    });

  } catch (error) {
    console.error('Error adding drug to patient:', error);
    res.status(500).json({ error: 'Failed to add drug to patient' });
  }
});

// Remove drug from patient
app.delete('/api/patients/:id/drugs/:drugName', async (req, res) => {
  try {
    const { id: patientId, drugName } = req.params;

    if (!drugName) {
      return res.status(400).json({ error: 'Drug name is required' });
    }

    // Get current patient data
    const { data: patient, error: patientError } = await supabase
      .from('patients')
      .select('*')
      .eq('id', patientId)
      .single();

    if (patientError || !patient) {
      return res.status(404).json({ error: 'Patient not found' });
    }

    // Check if drug exists in patient's medications
    const currentMedications = patient.medications || [];
    const decodedDrugName = decodeURIComponent(drugName);

    if (!currentMedications.includes(decodedDrugName)) {
      return res.status(404).json({ error: 'Drug not found in patient medications' });
    }

    // Remove drug from medications array
    const updatedMedications = currentMedications.filter(med => med !== decodedDrugName);

    // Update patient in database
    const { data: updatedPatient, error: updateError } = await supabase
      .from('patients')
      .update({
        medications: updatedMedications,
        updated_at: new Date().toISOString()
      })
      .eq('id', patientId)
      .select()
      .single();

    if (updateError) {
      throw updateError;
    }

    res.json({
      patient: updatedPatient,
      summary: {
        removedDrug: decodedDrugName,
        removedSuccessfully: true,
        remainingMedications: updatedMedications.length
      }
    });

  } catch (error) {
    console.error('Error removing drug from patient:', error);
    res.status(500).json({ error: 'Failed to remove drug from patient' });
  }
});

// Main interaction checking endpoint
app.post('/api/check-interactions', async (req, res) => {
  try {
    // Validate request
    const { error: validationError, value } = checkInteractionsSchema.validate(req.body);
    if (validationError) {
      return res.status(400).json({
        error: 'Invalid request',
        details: validationError.details
      });
    }

    const { medications, patientId } = value;

    // Log the interaction check if patient ID provided
    if (patientId) {
      await supabase
        .from('interaction_logs')
        .insert({
          patient_id: patientId,
          checked_meds: medications,
          results: null // Will be updated after processing
        });
    }

    // Get all unique pairs of medications
    const pairs = [];
    for (let i = 0; i < medications.length; i++) {
      for (let j = i + 1; j < medications.length; j++) {
        pairs.push([medications[i], medications[j]]);
      }
    }

    const results = [];

    // Check each pair for interactions
    for (const [drugA, drugB] of pairs) {
      try {
        // First, try to call the ML service
        const mlServiceUrl = process.env.ML_BASE || 'http://localhost:8000';
        const mlResponse = await fetch(`${mlServiceUrl}/interactions/check`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ drugA, drugB }),
          timeout: 5000
        });

        if (mlResponse.ok) {
          const mlResult = await mlResponse.json();
          results.push({
            drugA,
            drugB,
            severity: mlResult.severity || 'mild',
            description: mlResult.description || 'No significant interaction found',
            recommendation: mlResult.recommendation || 'Continue monitoring',
            sources: mlResult.sources || [],
            method: 'ml'
          });
        } else {
          throw new Error('ML service unavailable');
        }
      } catch (mlError) {
        console.log(`ML service error for ${drugA} + ${drugB}, using fallback:`, mlError.message);

        // Fallback: Use static rules or mark as unknown
        const fallbackResult = getFallbackInteraction(drugA, drugB);
        results.push({
          drugA,
          drugB,
          severity: fallbackResult.severity,
          description: fallbackResult.description,
          recommendation: fallbackResult.recommendation,
          sources: ['Internal Database'],
          method: 'fallback'
        });
      }
    }

    // Update interaction log with results if patient ID provided
    if (patientId) {
      await supabase
        .from('interaction_logs')
        .update({ results: results })
        .eq('patient_id', patientId)
        .eq('checked_meds', medications)
        .order('created_at', { ascending: false })
        .limit(1);
    }

    res.json({
      interactions: results,
      summary: {
        total_pairs: pairs.length,
        severe_interactions: results.filter(r => r.severity === 'severe').length,
        moderate_interactions: results.filter(r => r.severity === 'moderate').length,
        mild_interactions: results.filter(r => r.severity === 'mild').length
      }
    });

  } catch (error) {
    console.error('Error checking interactions:', error);
    res.status(500).json({ error: 'Failed to check interactions' });
  }
});

// Get interaction history for a patient
app.get('/api/patients/:id/interactions', async (req, res) => {
  try {
    const { id } = req.params;

    const { data, error } = await supabase
      .from('interaction_logs')
      .select('*')
      .eq('patient_id', id)
      .order('created_at', { ascending: false })
      .limit(10);

    if (error) throw error;
    res.json({ history: data });
  } catch (error) {
    console.error('Error fetching interaction history:', error);
    res.status(500).json({ error: 'Failed to fetch interaction history' });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 DDI Assistant API running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`🔗 ML Service: ${process.env.ML_BASE || 'http://localhost:8000'}`);
});