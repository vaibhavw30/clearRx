import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, CheckCircle, Info, Loader2, Plus, Search, X } from "lucide-react";
import { Patient, Interaction, InteractionSummary } from "@/types";
import { apiService } from "@/services/api";
import { Header } from "@/components/Header";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

// Minimal Health type used by the health check query
type Health = {
  status: string;
  services: Record<string, boolean>;
};

const Dashboard = () => {
  console.log('🎯 Dashboard component rendering...');
  console.log('🎯 Current window location:', window.location.href);
  console.log('🎯 Available environment variables:', {
    NODE_ENV: import.meta.env.NODE_ENV,
    VITE_API_BASE_URL: import.meta.env.VITE_API_BASE_URL,
    VITE_ML_BASE_URL: import.meta.env.VITE_ML_BASE_URL,
  });


  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [isCheckingInteractions, setIsCheckingInteractions] = useState(false);
  const [interactionResults, setInteractionResults] = useState<{
    interactions: Interaction[];
    summary: InteractionSummary;
  } | null>(null);

  // Drug addition states
  const [drugSearchQuery, setDrugSearchQuery] = useState('');
  const [drugSuggestions, setDrugSuggestions] = useState<any[]>([]);
  const [isSearchingDrugs, setIsSearchingDrugs] = useState(false);
  const [isAddingDrug, setIsAddingDrug] = useState(false);
  const [addDrugResults, setAddDrugResults] = useState<any>(null);

  const queryClient = useQueryClient();

  // Fetch patients
  const {
    data: patientsData,
    isLoading: patientsLoading,
    error: patientsError
  } = useQuery({
    queryKey: ['patients'],
    queryFn: async () => {
      console.log('🏥 Starting patients query...');
      try {
        const result = await apiService.getPatients();
        console.log('🏥 Patients query successful:', result);
        return result;
      } catch (error) {
        console.error('🏥 Patients query failed:', error);
        throw error;
      }
    },
    retry: 2,
    onError: (error) => {
      console.error('🏥 Patients query onError callback:', error);
    },
    onSuccess: (data) => {
      console.log('🏥 Patients query onSuccess callback:', data);
    }
  });

  // Health check
  const { data: healthData, error: healthError } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      console.log('❤️ Starting health check query...');
      try {
        const result = await apiService.healthCheck();
        console.log('❤️ Health check successful:', result);
        return result;
      } catch (error) {
        console.error('❤️ Health check failed:', error);
        throw error;
      }
    },
    refetchInterval: 30000, // Check every 30 seconds
    onError: (error) => {
      console.error('❤️ Health check onError callback:', error);
    },
    onSuccess: (data) => {
      console.log('❤️ Health check onSuccess callback:', data);
    }
  });

  // Log health to console (instead of showing UI alert). Typed to avoid TS errors.
  useEffect(() => {
    if (healthData) {
      const hd = healthData as Health;
      console.log('🩺 Health status (logged):', {
        status: hd.status,
        mlService: !!hd.services?.sentence_transformers,
        openai: !!hd.services?.openai,
      });
    }
  }, [healthData]);

  // Check interactions mutation
  const checkInteractionsMutation = useMutation({
    mutationFn: ({ medications, patientId }: { medications: string[]; patientId?: string }) =>
      apiService.checkInteractions(medications, patientId),
    onSuccess: (data) => {
      setInteractionResults(data);
      setIsCheckingInteractions(false);
      toast.success("Interaction check completed");
    },
    onError: (error) => {
      console.error('Error checking interactions:', error);
      setIsCheckingInteractions(false);
      toast.error("Failed to check interactions");
    }
  });

  const handlePatientSelect = (patientId: string) => {
    const patientsList = (patientsData as { patients: Patient[] } | undefined)?.patients;
    const patient = patientsList?.find(p => p.id === patientId);
    setSelectedPatient(patient || null);
    setInteractionResults(null); // Clear previous results
    setAddDrugResults(null); // Clear drug addition results
  };

  // Drug search functionality
  const handleDrugSearch = async (query: string) => {
    if (query.length < 2) {
      setDrugSuggestions([]);
      return;
    }

    setIsSearchingDrugs(true);
    try {
      const result = await apiService.searchDrugs(query);
      setDrugSuggestions(result.suggestions || []);
    } catch (error) {
      console.error('Drug search failed:', error);
      toast.error('Failed to search drugs');
      setDrugSuggestions([]);
    } finally {
      setIsSearchingDrugs(false);
    }
  };

  // Add drug to patient
  const handleAddDrug = async (drugName: string, rxcui?: string) => {
    if (!selectedPatient) {
      toast.error('Please select a patient first');
      return;
    }

    setIsAddingDrug(true);
    try {
      const result = await apiService.addDrugToPatient(selectedPatient.id, drugName, rxcui);
      setAddDrugResults(result);

      // Update the selected patient with new medication list
      setSelectedPatient(result.patient);

      // Refresh patients data
  // Refresh patients data
  queryClient.invalidateQueries({ queryKey: ['patients'] });

      toast.success(`${drugName} added successfully`);

      // Clear search
      setDrugSearchQuery('');
      setDrugSuggestions([]);

    } catch (error) {
      console.error('Failed to add drug:', error);
      toast.error('Failed to add drug to patient');
    } finally {
      setIsAddingDrug(false);
    }
  };

  const handleRemoveDrug = async (drugName: string) => {
    if (!selectedPatient) {
      toast.error('Please select a patient first');
      return;
    }

    try {
      const result = await apiService.removeDrugFromPatient(selectedPatient.id, drugName);

      // Update the selected patient with removed medication
      setSelectedPatient(result.patient);

      // Refresh patients data
      queryClient.invalidateQueries({ queryKey: ['patients'] });

      // Clear any previous results
      setAddDrugResults(null);
      setInteractionResults(null);

      toast.success(`${drugName} removed successfully`);
    } catch (error) {
      console.error('Error removing drug:', error);
      toast.error('Failed to remove medication');
    }
  };

  const handleCheckInteractions = async () => {
    if (!selectedPatient || selectedPatient.medications.length < 2) {
      toast.error("Please select a patient with at least 2 medications");
      return;
    }

    setIsCheckingInteractions(true);
    checkInteractionsMutation.mutate({
      medications: selectedPatient.medications,
      patientId: selectedPatient.id
    });
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'severe':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'moderate':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'mild':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'severe':
        return <AlertCircle className="h-4 w-4" />;
      case 'moderate':
        return <Info className="h-4 w-4" />;
      case 'mild':
        return <CheckCircle className="h-4 w-4" />;
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  console.log('🎯 Dashboard state check:', {
    patientsLoading,
    patientsError: patientsError?.message || patientsError,
    patientsData,
    healthError: healthError?.message || healthError,
    healthData
  });

  if (patientsError) {
    console.error('🎯 Showing patients error UI due to:', patientsError);
    return (
      <div className="min-h-screen bg-gray-50">
        <Header />
        <div className="container mx-auto px-4 py-8">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Failed to connect to the backend service. Please ensure the API server is running.
              <br />
              <strong>Debug Info:</strong> {String(patientsError)}
            </AlertDescription>
          </Alert>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <div className="container mx-auto px-4 py-8">
        {/* Service Status: logged to console instead of shown in UI */}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Patient Selection */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle>Select Patient</CardTitle>
                <CardDescription>
                  Choose a patient to analyze their medications
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {patientsLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin" />
                  </div>
                ) : (
                  <Select onValueChange={handlePatientSelect}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a patient" />
                    </SelectTrigger>
                    <SelectContent>
                      {((patientsData as { patients: Patient[] } | undefined)?.patients || []).map((patient) => (
                        <SelectItem key={patient.id} value={patient.id}>
                          {patient.name} ({patient.age}y, {patient.gender})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}

                {selectedPatient && (
                  <div className="space-y-3">
                    <div>
                      <h4 className="font-medium">Patient Information</h4>
                      <p className="text-sm text-gray-600">
                        {selectedPatient.name}, {selectedPatient.age} years old
                      </p>
                      {selectedPatient.conditions && selectedPatient.conditions.length > 0 && (
                        <div className="mt-2">
                          <p className="text-xs font-medium text-gray-700">Conditions:</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {selectedPatient.conditions.map((condition, index) => (
                              <Badge key={index} variant="outline" className="text-xs">
                                {condition}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    <Separator />

                    <div>
                      <h4 className="font-medium">Current Medications</h4>
                      <div className="space-y-2 mt-2">
                        {selectedPatient.medications.map((medication, index) => (
                          <div key={index} className="flex items-center justify-between gap-2">
                            <Badge variant="secondary">{medication}</Badge>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleRemoveDrug(medication)}
                              className="h-6 w-6 p-0 hover:bg-red-100 hover:text-red-600"
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>

                    <Separator />

                    <div>
                      <h4 className="font-medium flex items-center gap-2">
                        <Plus className="h-4 w-4" />
                        Add New Medication
                      </h4>
                      <div className="space-y-3 mt-2">
                        <div className="relative">
                          <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                          <Input
                            placeholder="Search for medication..."
                            value={drugSearchQuery}
                            onChange={(e) => {
                              setDrugSearchQuery(e.target.value);
                              handleDrugSearch(e.target.value);
                            }}
                            className="pl-10"
                          />
                          {isSearchingDrugs && (
                            <Loader2 className="absolute right-3 top-3 h-4 w-4 animate-spin" />
                          )}
                        </div>

                        {drugSuggestions.length > 0 && (
                          <div className="max-h-48 overflow-y-auto border rounded-md bg-white">
                            {drugSuggestions.map((drug, index) => (
                              <div
                                key={index}
                                className="p-3 hover:bg-gray-50 cursor-pointer border-b last:border-b-0"
                                onClick={() => handleAddDrug(drug.name, drug.rxcui)}
                              >
                                <div className="font-medium">{drug.name}</div>
                                <div className="text-sm text-gray-500">RxCUI: {drug.rxcui}</div>
                                {drug.score && (
                                  <div className="text-xs text-gray-400">Match: {drug.score}</div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}

                        {isAddingDrug && (
                          <div className="flex items-center gap-2 text-sm text-gray-600">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Adding medication and checking interactions...
                          </div>
                        )}
                      </div>
                    </div>

                    <Button
                      onClick={handleCheckInteractions}
                      disabled={isCheckingInteractions || selectedPatient.medications.length < 2}
                      className="w-full"
                    >
                      {isCheckingInteractions ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Checking Interactions...
                        </>
                      ) : (
                        'Check Drug Interactions'
                      )}
                    </Button>

                    {selectedPatient.medications.length < 2 && (
                      <p className="text-xs text-gray-500">
                        At least 2 medications required for interaction checking
                      </p>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Results */}
          <div className="lg:col-span-2">
            {addDrugResults ? (
              <div className="space-y-6">
                {/* Drug Addition Summary */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Plus className="h-5 w-5" />
                      Drug Added: {addDrugResults.summary?.newDrug}
                    </CardTitle>
                    <CardDescription>
                      Interaction analysis for newly added medication
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                          ✓
                        </div>
                        <div className="text-sm text-gray-600">Added Successfully</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">
                          {addDrugResults.summary?.totalChecks || 0}
                        </div>
                        <div className="text-sm text-gray-600">Interactions Checked</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                          {addDrugResults.summary?.interactionsFound || 0}
                        </div>
                        <div className="text-sm text-gray-600">Risks Found</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Interaction Results for New Drug */}
                {addDrugResults.interactionResults?.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Interaction Analysis</CardTitle>
                      <CardDescription>
                        Potential interactions with {addDrugResults.summary?.newDrug}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {addDrugResults.interactionResults.map((interaction: any, index: number) => (
                          <div
                            key={index}
                            className={`p-4 rounded-lg border ${getSeverityColor(interaction.severity)}`}
                          >
                            <div className="flex items-start gap-3">
                              {getSeverityIcon(interaction.severity)}
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  <h4 className="font-medium">
                                    {interaction.existingDrug} + {interaction.newDrug}
                                  </h4>
                                  <Badge
                                    variant="outline"
                                    className={getSeverityColor(interaction.severity)}
                                  >
                                    {interaction.severity?.toUpperCase() || 'UNKNOWN'}
                                  </Badge>
                                  <Badge variant="outline" className="text-xs">
                                    {interaction.method || 'enhanced_ai'}
                                  </Badge>
                                </div>

                                <p className="text-sm mb-2">{interaction.description}</p>

                                <div className="text-sm">
                                  <strong>Recommendation:</strong> {interaction.recommendation}
                                </div>

                                {interaction.confidence && (
                                  <div className="text-xs text-gray-600 mt-1">
                                    <strong>Confidence:</strong> {(interaction.confidence * 100).toFixed(1)}%
                                  </div>
                                )}

                                {interaction.riskFactors && interaction.riskFactors.length > 0 && (
                                  <div className="text-xs text-gray-600 mt-2">
                                    <strong>Risk Factors:</strong> {interaction.riskFactors.join(', ')}
                                  </div>
                                )}

                                {interaction.monitoringAdvice && (
                                  <div className="text-xs text-gray-600 mt-1">
                                    <strong>Monitoring:</strong> {interaction.monitoringAdvice}
                                  </div>
                                )}

                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Clear Results Button */}
                <div className="flex justify-end">
                  <Button
                    variant="outline"
                    onClick={() => setAddDrugResults(null)}
                  >
                    Clear Results
                  </Button>
                </div>
              </div>
            ) : interactionResults ? (
              <div className="space-y-6">
                {/* Summary */}
                <Card>
                  <CardHeader>
                    <CardTitle>Interaction Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-gray-900">
                          {interactionResults.summary.total_pairs}
                        </div>
                        <div className="text-sm text-gray-600">Total Pairs</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                          {interactionResults.summary.severe_interactions}
                        </div>
                        <div className="text-sm text-gray-600">Severe</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-yellow-600">
                          {interactionResults.summary.moderate_interactions}
                        </div>
                        <div className="text-sm text-gray-600">Moderate</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                          {interactionResults.summary.mild_interactions}
                        </div>
                        <div className="text-sm text-gray-600">Mild</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Detailed Results */}
                <Card>
                  <CardHeader>
                    <CardTitle>Detailed Interactions</CardTitle>
                    <CardDescription>
                      {interactionResults.interactions.length} drug pair(s) analyzed
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {interactionResults.interactions.map((interaction, index) => (
                        <div
                          key={index}
                          className={`p-4 rounded-lg border ${getSeverityColor(interaction.severity)}`}
                        >
                          <div className="flex items-start gap-3">
                            {getSeverityIcon(interaction.severity)}
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-2">
                                <h4 className="font-medium">
                                  {interaction.drugA} + {interaction.drugB}
                                </h4>
                                <Badge
                                  variant="outline"
                                  className={getSeverityColor(interaction.severity)}
                                >
                                  {interaction.severity.toUpperCase()}
                                </Badge>
                                <Badge variant="outline" className="text-xs">
                                  {interaction.method || 'enhanced_ai'}
                                </Badge>
                              </div>

                              <p className="text-sm mb-2">{interaction.description}</p>

                              <div className="text-sm">
                                <strong>Recommendation:</strong> {interaction.recommendation}
                              </div>

                              {interaction.confidence && (
                                <div className="text-xs text-gray-600 mt-1">
                                  <strong>Confidence:</strong> {(interaction.confidence * 100).toFixed(1)}%
                                </div>
                              )}

                              {interaction.sources.length > 0 && (
                                <div className="text-xs text-gray-600 mt-2">
                                  <strong>Sources:</strong> {interaction.sources.join(', ')}
                                </div>
                              )}

                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardContent className="flex items-center justify-center py-16">
                  <div className="text-center">
                    <Info className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      No Analysis Yet
                    </h3>
                    <p className="text-gray-600">
                      Select a patient and click "Check Drug Interactions" to see results
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;