// Backend API integration for AI detection

// Configuration
const BACKEND_API_URL =
  (typeof import.meta !== 'undefined' && (import.meta as any).env && (import.meta as any).env.VITE_BACKEND_API_URL) ||
  (typeof process !== 'undefined' && (process as any).env && (process as any).env.VITE_BACKEND_API_URL) ||
  'http://localhost:5000';

export interface DetectionResult {
  personCount: number;
  deviceDetected: boolean;
  confidence: number;
  timestamp: number;
  headPose?: {
    direction: string;
    confidence: number;
  };
  bodyVisibility?: {
    visible: boolean;
    confidence: number;
  };
  peopleLocations?: Array<{
    person_id: number;
    bbox: { x1: number; y1: number; x2: number; y2: number };
    confidence: number;
  }>;
  modelInfo?: {
    multiPersonModel: string;
    headPoseModel: string;
    bodyVisibilityModel: string;
  };
}

export interface ViolationAlert {
  type: 'multiple_persons' | 'device_detected' | 'head_pose_violation' | 'body_visibility_violation';
  message: string;
  confidence: number;
  timestamp: number;
}

class AIDetectionService {
  private isInitialized = false;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private backendHealthy = false;

  constructor() {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      console.log('🚀 [AI Detection] Initializing with backend API...');
      console.log('🌐 [AI Detection] Backend URL:', BACKEND_API_URL);
      
      // Check backend health
      console.log('🔍 [AI Detection] Checking backend health...');
      const healthResponse = await fetch(`${BACKEND_API_URL}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (healthResponse.ok) {
        const healthData = await healthResponse.json();
        this.backendHealthy = true;
        console.log('✅ [AI Detection] Backend is healthy:', healthData);
      } else {
        console.error('❌ [AI Detection] Backend health check failed:', healthResponse.status);
        throw new Error(`Backend health check failed: ${healthResponse.status}`);
      }

      // Get system status
      console.log('📊 [AI Detection] Getting system status...');
      const statusResponse = await fetch(`${BACKEND_API_URL}/api/system/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (statusResponse.ok) {
        const status = await statusResponse.json();
        console.log('✅ [AI Detection] System status:', status);
        console.log(`📈 [AI Detection] Models loaded: ${status.system.models_loaded}`);
        console.log('🎯 [AI Detection] Capabilities:', status.capabilities);
        
        if (status.capabilities.unified_inference) {
          console.log('🔥 [AI Detection] Unified inference system ready!');
        } else {
          console.warn('⚠️ [AI Detection] Unified inference not ready, using individual models');
        }
      } else {
        console.warn('⚠️ [AI Detection] Could not get system status');
      }

      this.isInitialized = true;
      console.log('🎉 [AI Detection] Service initialized successfully!');
    } catch (error) {
      console.error('❌ [AI Detection] Failed to initialize:', error);
      this.backendHealthy = false;
      console.warn('🔄 [AI Detection] Will use fallback detection');
      this.isInitialized = true; // Still mark as initialized for fallback
    }
  }

  private captureFrame(videoElement: HTMLVideoElement): string {
    // Set canvas size to match video
    this.canvas.width = videoElement.videoWidth;
    this.canvas.height = videoElement.videoHeight;
    
    // Draw video frame to canvas
    this.ctx.drawImage(videoElement, 0, 0);
    
    // Convert to base64
    return this.canvas.toDataURL('image/jpeg', 0.8);
  }

  private async callBackendAPI(endpoint: string, imageData: string): Promise<any> {
    try {
      const response = await fetch(`${BACKEND_API_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_b64: imageData,
          user_id: Date.now().toString() // Simple user ID for now
        })
      });

      if (!response.ok) {
        throw new Error(`API call failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`Backend API error (${endpoint}):`, error);
      throw error;
    }
  }

  private async runFallbackDetection(videoElement: HTMLVideoElement): Promise<DetectionResult> {
    // Simple client-side fallback when backend is unavailable
    console.warn('Using fallback detection - limited functionality');
    
    const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const personCount = this.simpleFaceDetection(imageData);
    
    return {
      personCount,
      deviceDetected: false, // Cannot detect devices without backend
      confidence: 0.5, // Lower confidence for fallback
      timestamp: Date.now()
    };
  }

  private simpleFaceDetection(imageData: ImageData): number {
    // Very basic face detection using skin tone analysis
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    
    let skinPixels = 0;
    const totalPixels = width * height;
    
    for (let i = 0; i < data.length; i += 16) { // Sample every 4th pixel
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // Basic skin tone detection
      if (r > 95 && g > 40 && b > 20 && r > g && r > b) {
        skinPixels++;
      }
    }
    
    const skinRatio = skinPixels / (totalPixels / 16);
    
    // Estimate person count based on skin pixel ratio
    if (skinRatio > 0.15) return 2; // Likely multiple people
    if (skinRatio > 0.08) return 1; // Likely one person
    return 0; // No clear person detected
  }

  async detectViolations(videoElement: HTMLVideoElement): Promise<DetectionResult> {
    console.log('🔍 [AI Detection] Starting violation detection...');
    
    if (!this.isInitialized) {
      console.log('🔄 [AI Detection] Not initialized, initializing now...');
      await this.initialize();
    }

    if (!this.backendHealthy) {
      console.warn('⚠️ [AI Detection] Backend unhealthy, using fallback');
      return this.runFallbackDetection(videoElement);
    }

    try {
      console.log('📷 [AI Detection] Capturing video frame...');
      // Capture current frame
      const imageData = this.captureFrame(videoElement);
      console.log('✅ [AI Detection] Frame captured, size:', imageData.length);
      
      console.log('🌐 [AI Detection] Sending to backend API...');
      // Call unified detection endpoint for comprehensive analysis
      const result = await this.callBackendAPI('/api/detections/unified', imageData);
      console.log('✅ [AI Detection] Backend response:', result);
      
      // Extract detection results
      const multiPersonData = result.detections?.multi_person || {};
      const headPoseData = result.detections?.head_pose || {};
      const bodyVisibilityData = result.detections?.body_visibility || {};
      
      console.log('📈 [AI Detection] Extracted data:', {
        multiPerson: multiPersonData,
        headPose: headPoseData,
        bodyVisibility: bodyVisibilityData
      });
      
      // Map backend results to our interface
      const detectionResult: DetectionResult = {
        personCount: multiPersonData.num_people || 0,
        deviceDetected: headPoseData.violation || false, // Head pose violations often indicate device usage
        confidence: Math.max(
          multiPersonData.confidence || 0,
          headPoseData.confidence || 0,
          bodyVisibilityData.confidence || 0
        ),
        timestamp: Date.now(),
        headPose: {
          direction: headPoseData.direction || 'unknown',
          confidence: headPoseData.confidence || 0
        },
        bodyVisibility: {
          visible: !bodyVisibilityData.violation,
          confidence: bodyVisibilityData.confidence || 0
        },
        peopleLocations: multiPersonData.people_locations || [],
        modelInfo: {
          multiPersonModel: multiPersonData.model_type || 'unknown',
          headPoseModel: 'MobileNetV2',
          bodyVisibilityModel: 'Custom'
        }
      };
      
      console.log('🎉 [AI Detection] Final result:', detectionResult);
      console.log('✅ [AI Detection] Backend AI detection completed:', {
        people: detectionResult.personCount,
        confidence: detectionResult.confidence,
        headDirection: detectionResult.headPose?.direction,
        modelInfo: detectionResult.modelInfo,
        violations: result.violations?.total_violations || 0
      });
      
      return detectionResult;
      
    } catch (error) {
      console.error('Backend detection failed, using fallback:', error);
      this.backendHealthy = false;
      return this.runFallbackDetection(videoElement);
    }
  }

  async analyzeForViolations(videoElement: HTMLVideoElement): Promise<ViolationAlert[]> {
    const result = await this.detectViolations(videoElement);
    const violations: ViolationAlert[] = [];

    // Multiple persons violation
    if (result.personCount > 1) {
      violations.push({
        type: 'multiple_persons',
        message: `${result.personCount} people detected in exam area (Confidence: ${Math.round(result.confidence * 100)}%)`,
        confidence: result.confidence,
        timestamp: result.timestamp
      });
    }

    // Head pose violation (looking away from screen)
    if (result.headPose && result.headPose.direction !== 'center' && result.headPose.direction !== 'unknown') {
      violations.push({
        type: 'head_pose_violation',
        message: `Student looking ${result.headPose.direction} (Confidence: ${Math.round(result.headPose.confidence * 100)}%)`,
        confidence: result.headPose.confidence,
        timestamp: result.timestamp
      });
    }

    // Body visibility violation
    if (result.bodyVisibility && !result.bodyVisibility.visible) {
      violations.push({
        type: 'body_visibility_violation',
        message: `Student body not fully visible (Confidence: ${Math.round(result.bodyVisibility.confidence * 100)}%)`,
        confidence: result.bodyVisibility.confidence,
        timestamp: result.timestamp
      });
    }

    // Device detection (based on head pose patterns)
    if (result.deviceDetected) {
      violations.push({
        type: 'device_detected',
        message: `Potential device usage detected (Confidence: ${Math.round(result.confidence * 100)}%)`,
        confidence: result.confidence,
        timestamp: result.timestamp
      });
    }

    return violations;
  }

  async getSystemStatus(): Promise<any> {
    if (!this.backendHealthy) {
      return {
        healthy: false,
        error: 'Backend not available',
        usingFallback: true
      };
    }

    try {
      const response = await fetch(`${BACKEND_API_URL}/api/system/status`);
      if (response.ok) {
        return await response.json();
      }
      throw new Error('Status check failed');
    } catch (error) {
      console.error('System status check failed:', error);
      return {
        healthy: false,
        error: error.message,
        usingFallback: true
      };
    }
  }

  dispose(): void {
    // No models to dispose of since we're using backend API
    console.log('AI detection service disposed');
  }
}

export const aiDetectionService = new AIDetectionService();