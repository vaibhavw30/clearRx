export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  conditions?: string[];
  medications: string[]; // Array of medication names
  created_at?: string;
  updated_at?: string;
}

export interface Medication {
  id: string;
  name: string;
  dose_form: string;
}

export interface Drug {
  id: string;
  name: string;
  common_uses: string;
  side_effects: string[];
  warnings: string[];
}

export interface Interaction {
  drugA: string;
  drugB: string;
  severity: 'mild' | 'moderate' | 'severe';
  description: string;
  recommendation: string;
  sources: string[];
  confidence?: number;
  method: string;
}

export interface InteractionSummary {
  total_pairs: number;
  severe_interactions: number;
  moderate_interactions: number;
  mild_interactions: number;
}

export interface InteractionResult {
  interactions: Interaction[];
  summary: InteractionSummary;
}

// Legacy types for backward compatibility
export interface LegacyInteraction {
  id: string;
  drug_a_id: string;
  drug_b_id: string;
  drug_a_name: string;
  drug_b_name: string;
  severity: number; // 1=low, 2=moderate, 3=high
  description: string;
  mechanism?: string;
  recommended_action?: string;
  source_ref?: string;
}