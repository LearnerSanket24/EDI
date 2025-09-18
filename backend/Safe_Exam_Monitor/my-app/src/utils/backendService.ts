// Backend service integration utility
// Provides centralized access to Python Flask API endpoints

const BACKEND_API_URL = 'http://localhost:5000';

export interface BackendHealth {
  status: string;
  healthy: boolean;
}

export interface SystemStatus {
  system: {
    status: string;
    version: string;
    models_loaded: number;
    timestamp: string;
  };
  models: {
    unified_detection: any;
    head_pose_legacy: any;
  };
  capabilities: {
    multi_person_detection: boolean;
    head_pose_detection: boolean;
    body_visibility_detection: boolean;
    unified_inference: boolean;
  };
}

export interface AlertRequest {
  student: string;
  violation: string;
  timestamp: string;
}

class BackendService {
  private healthyStatus: boolean = false;
  private lastHealthCheck: number = 0;
  private healthCheckInterval: number = 30000; // 30 seconds

  constructor() {
    // Initial health check
    this.checkHealth();
  }

  private async makeRequest(endpoint: string, options: RequestInit = {}): Promise<Response> {
    const url = `${BACKEND_API_URL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response;
    } catch (error) {
      console.error(`Backend API request failed (${endpoint}):`, error);
      throw error;
    }
  }

  async checkHealth(): Promise<BackendHealth> {
    try {
      const response = await this.makeRequest('/api/health');
      const data = await response.json();
      
      this.healthyStatus = data.status === 'ok';
      this.lastHealthCheck = Date.now();
      
      return {
        status: data.status,
        healthy: this.healthyStatus
      };
    } catch (error) {
      this.healthyStatus = false;
      console.error('Backend health check failed:', error);
      return {
        status: 'error',
        healthy: false
      };
    }
  }

  async getSystemStatus(): Promise<SystemStatus | null> {
    try {
      const response = await this.makeRequest('/api/system/status');
      return await response.json();
    } catch (error) {
      console.error('Failed to get system status:', error);
      return null;
    }
  }

  async detectHeadPose(imageBase64: string, userId?: string): Promise<any> {
    try {
      const response = await this.makeRequest('/api/detections/head_pose', {
        method: 'POST',
        body: JSON.stringify({
          image_b64: imageBase64,
          user_id: userId || Date.now().toString()
        })
      });
      return await response.json();
    } catch (error) {
      console.error('Head pose detection failed:', error);
      throw error;
    }
  }

  async detectMultiplePerson(imageBase64: string, userId?: string): Promise<any> {
    try {
      const response = await this.makeRequest('/api/detections/multi_person', {
        method: 'POST',
        body: JSON.stringify({
          image_b64: imageBase64,
          user_id: userId || Date.now().toString()
        })
      });
      return await response.json();
    } catch (error) {
      console.error('Multi-person detection failed:', error);
      throw error;
    }
  }

  async detectBodyVisibility(imageBase64: string, userId?: string): Promise<any> {
    try {
      const response = await this.makeRequest('/api/detections/body_visibility', {
        method: 'POST',
        body: JSON.stringify({
          image_b64: imageBase64,
          user_id: userId || Date.now().toString()
        })
      });
      return await response.json();
    } catch (error) {
      console.error('Body visibility detection failed:', error);
      throw error;
    }
  }

  async runUnifiedDetection(imageBase64: string, userId?: string): Promise<any> {
    try {
      const response = await this.makeRequest('/api/detections/unified', {
        method: 'POST',
        body: JSON.stringify({
          image_b64: imageBase64,
          user_id: userId || Date.now().toString()
        })
      });
      return await response.json();
    } catch (error) {
      console.error('Unified detection failed:', error);
      throw error;
    }
  }

  async sendAlert(alertData: AlertRequest): Promise<boolean> {
    try {
      const response = await this.makeRequest('/api/alert', {
        method: 'POST',
        body: JSON.stringify(alertData)
      });
      
      const result = await response.json();
      return result.sent === true;
    } catch (error) {
      console.error('Failed to send alert:', error);
      return false;
    }
  }

  isHealthy(): boolean {
    // Check if we need a health check update
    const now = Date.now();
    if (now - this.lastHealthCheck > this.healthCheckInterval) {
      // Trigger async health check but return current status
      this.checkHealth();
    }
    return this.healthyStatus;
  }

  getApiUrl(): string {
    return BACKEND_API_URL;
  }
}

// Create and export singleton instance
export const backendService = new BackendService();

// Helper function to convert video frame to base64
export function captureVideoFrame(videoElement: HTMLVideoElement): string {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  
  ctx.drawImage(videoElement, 0, 0);
  
  return canvas.toDataURL('image/jpeg', 0.8);
}

// Export types for use in components
export type { BackendHealth, SystemStatus, AlertRequest };
