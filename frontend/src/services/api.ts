// API service for connecting to the Express backend
import { createSSEParser } from '@/services/sse';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001';

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  conditions: string[];
  medications: string[];
  created_at?: string;
  updated_at?: string;
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

export interface Citation {
  source_doc_id: string;
  section?: string | null;
  url?: string | null;
}

class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
    console.log('🔗 ApiService initialized with baseUrl:', this.baseUrl);
    console.log('🔗 Environment variables:', {
      VITE_API_BASE_URL: import.meta.env.VITE_API_BASE_URL,
      VITE_ML_BASE_URL: import.meta.env.VITE_ML_BASE_URL,
    });

    // Test basic connectivity
    this.testConnectivity();
  }

  private async testConnectivity() {
    console.log('🧪 Testing API connectivity...');
    try {
      const testUrl = `${this.baseUrl}/api/health`;
      console.log('🧪 Testing URL:', testUrl);

      const response = await fetch(testUrl, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      console.log('🧪 Connectivity test result:', {
        url: testUrl,
        status: response.status,
        ok: response.ok,
        headers: Object.fromEntries(response.headers.entries())
      });

      if (response.ok) {
        const data = await response.json();
        console.log('🧪 Connectivity test successful:', data);
      } else {
        console.error('🧪 Connectivity test failed - response not ok');
      }
    } catch (error) {
      console.error('🧪 Connectivity test failed with exception:', error);
    }
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    console.log('🚀 Making API request:', {
      method: options?.method || 'GET',
      url,
      headers: options?.headers,
      body: options?.body,
    });

    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      console.log('📡 API response received:', {
        url,
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
        headers: Object.fromEntries(response.headers.entries()),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        console.error('❌ API request failed - Response not OK:', {
          url,
          status: response.status,
          statusText: response.statusText,
          errorData,
        });
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('✅ API request successful:', {
        url,
        dataKeys: Object.keys(data),
        dataPreview: JSON.stringify(data).substring(0, 200) + '...',
      });

      return data;
    } catch (error) {
      console.error('💥 API request failed with exception:', {
        url,
        error: error instanceof Error ? error.message : error,
        stack: error instanceof Error ? error.stack : undefined,
      });
      throw error;
    }
  }

  async streamQuery(
    query: string,
    handlers: {
      onToken: (t: string) => void;
      onCitations?: (c: Citation[]) => void;
      onDone?: () => void;
      onError?: (e: unknown) => void;
    },
  ): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!response.ok || !response.body) {
        handlers.onError?.(new Error(`stream request failed: ${response.status}`));
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const parser = createSSEParser();
      let finished = false;
      while (!finished) {
        const { value, done } = await reader.read();
        if (done) break;
        for (const ev of parser.feed(decoder.decode(value, { stream: true }))) {
          if (ev.type === 'token') handlers.onToken(ev.data);
          else if (ev.type === 'citations') handlers.onCitations?.(ev.data as Citation[]);
          else if (ev.type === 'done') { handlers.onDone?.(); finished = true; }
        }
      }
      if (!finished) handlers.onDone?.();
    } catch (e) {
      handlers.onError?.(e);
    }
  }

  // Health check
  async healthCheck(): Promise<{ status: string; services: Record<string, boolean> }> {
    return this.request('/api/health');
  }

  // Patient endpoints
  async getPatients(): Promise<{ patients: Patient[] }> {
    return this.request('/api/patients');
  }

  async getPatient(id: string): Promise<{ patient: Patient }> {
    return this.request(`/api/patients/${id}`);
  }

  async getPatientInteractionHistory(id: string): Promise<{ history: any[] }> {
    return this.request(`/api/patients/${id}/interactions`);
  }

  // Drug endpoints
  async getDrugs(): Promise<{ drugs: Drug[] }> {
    return this.request('/api/drugs');
  }

  async getDrug(name: string): Promise<{ drug: Drug }> {
    return this.request(`/api/drugs/${encodeURIComponent(name)}`);
  }

  // Main interaction checking
  async checkInteractions(
    medications: string[],
    patientId?: string
  ): Promise<InteractionResult> {
    return this.request('/api/check-interactions', {
      method: 'POST',
      body: JSON.stringify({
        medications,
        patientId,
      }),
    });
  }

  // Search for drugs using RxNorm API
  async searchDrugs(query: string): Promise<{ suggestions: any[] }> {
    return this.request(`/api/drugs/search/${encodeURIComponent(query)}`);
  }

  // Get detailed drug information
  async getDrugDetails(rxcui: string): Promise<{ drug: any; adverseEvents: any[] }> {
    return this.request(`/api/drugs/details/${rxcui}`);
  }

  // Add drug to patient and check for interactions
  async addDrugToPatient(
    patientId: string,
    drugName: string,
    rxcui?: string
  ): Promise<{ patient: Patient; interactionResults: any[]; summary: any }> {
    return this.request(`/api/patients/${patientId}/drugs`, {
      method: 'POST',
      body: JSON.stringify({
        drugName,
        rxcui,
      }),
    });
  }

  // Remove drug from patient
  async removeDrugFromPatient(
    patientId: string,
    drugName: string
  ): Promise<{ patient: Patient; summary: any }> {
    return this.request(`/api/patients/${patientId}/drugs/${encodeURIComponent(drugName)}`, {
      method: 'DELETE',
    });
  }
}

export const apiService = new ApiService();
export default apiService;