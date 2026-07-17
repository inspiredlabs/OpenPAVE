// ./aspectRatioCamera.js - General Aspect-Ratio Aware Camera with Zoom
import * as THREE from 'three';
import { useContext } from './simpleContext.svelte.js';

// General positions - looking toward target (default in positive Z):
// const BASE_POSITIONS = {
//   WIDESCREEN: {
//     position: { x: 0, y: 1.6, z: 21.0 },
//     lookAt: { x: 0, y: 1.6, z: 0 }
//   },
//   PORTRAIT: {
//     position: { x: 0, y: 1.6, z: 60 },
//     lookAt: { x: 0, y: 1.6, z: 0 }
//   }
// };

const BASE_POSITIONS = {
  WIDESCREEN: {
    position: { x: 0, y: 1.6, z: 19.8 },
    lookAt: { x: 0, y: 1.6, z: 0 }
  },
  PORTRAIT: {
    position: { x: 0, y: 1.6, z: 54.0 },
    lookAt: { x: 0, y: 1.6, z: 0 }
  }
};

const ASPECT_RATIO_THRESHOLD = 1.0;

/**
 * Calculate FOV needed to keep viewed width visible
 */
function calculateFOVForViewedWidth(viewedWidth, distance, aspectRatio) {
  const horizontalFOV = 2 * Math.atan((viewedWidth / 2) / distance);
  const verticalFOV = horizontalFOV / aspectRatio;
  return THREE.MathUtils.radToDeg(verticalFOV);
}

/**
 * Smooth interpolation between camera configs
 */
function lerpCameraConfig(config1, config2, t) {
  const lerp = (a, b, t) => a + (b - a) * t;
  return {
    position: {
      x: lerp(config1.position.x, config2.position.x, t),
      y: lerp(config1.position.y, config2.position.y, t),
      z: lerp(config1.position.z, config2.position.z, t)
    },
    lookAt: {
      x: lerp(config1.lookAt.x, config2.lookAt.x, t),
      y: lerp(config1.lookAt.y, config2.lookAt.y, t),
      z: lerp(config1.lookAt.z, config2.lookAt.z, t)
    }
  };
}

/**
 * Calculate camera configuration for aspect ratio and zoom
 */
function calculateCameraConfig(aspectRatio, zoom = 1.0, viewedWidth = 1.0) {  // Default viewedWidth for BoxCollider, adjust as needed
  let config;
  
  if (aspectRatio >= ASPECT_RATIO_THRESHOLD) {
    config = { ...BASE_POSITIONS.WIDESCREEN };
  } else {
    const clampedRatio = Math.max(0.5, Math.min(1.0, aspectRatio));
    const t = (clampedRatio - 0.5) / (0.5);
    config = lerpCameraConfig(BASE_POSITIONS.PORTRAIT, BASE_POSITIONS.WIDESCREEN, t);
  }
  
  // Apply zoom: scale distance from lookAt
  const basePosition = new THREE.Vector3(
    config.position.x,
    config.position.y,
    config.position.z
  );
  const lookAtPos = new THREE.Vector3(
    config.lookAt.x,
    config.lookAt.y,
    config.lookAt.z
  );
  
  const direction = basePosition.clone().sub(lookAtPos);
  const baseDistance = direction.length();
  const newDistance = baseDistance * zoom;
  
  direction.normalize().multiplyScalar(newDistance);
  const newPosition = lookAtPos.clone().add(direction);
  
  config.position = {
    x: newPosition.x,
    y: newPosition.y,
    z: newPosition.z
  };
  
  // Calculate FOV based on viewed width at the new distance
  const requiredFOV = calculateFOVForViewedWidth(viewedWidth, newDistance, aspectRatio);
  
  return {
    ...config,
    fov: requiredFOV
  };
}

/**
 * General aspect-ratio aware camera attachment with zoom
 */
export default function aspectRatioCamera(options = {}) {
  return (canvas) => {
    const context = useContext();
    
    if (!context) {
      console.warn('[AspectRatioCamera] No context available');
      return () => {};
    }
    
    // Configuration
    const config = {
      minFOV: options.minFOV || 30,
      maxFOV: options.maxFOV || 120,
      smoothingFactor: options.smoothingFactor || 0.05,
      zoom: options.zoom || 1.0,
      viewedWidth: options.viewedWidth,  // Width to fit, e.g., for BoxCollider total width
      direction: options.direction || 1,  // 1 for looking +Z, -1 for -Z
      debug: options.debug || false,
      ...options
    };
    
    let lastWidth = 0;
    let lastHeight = 0;
    let lastZoom = config.zoom;
    let targetCameraConfig = null;
    let currentCameraConfig = null;
    let targetFOV = 60;
    let currentFOV = 60;
    
    /**
     * Apply camera positioning
     */
    function applyCameraPosition() {
      const currentWidth = context.canvasWidth || canvas.clientWidth;
      const currentHeight = context.canvasHeight || canvas.clientHeight;
      
      const activeCamera = context.getCamera();
      // if (!activeCamera || !activeCamera.isPerspectiveCamera) {
      //   if (config.debug) console.log('[AspectRatioCamera] No perspective camera available');
      //   return;
      // }
      
      // Smooth transitions if we have a target
      if (currentCameraConfig && targetCameraConfig) {
        currentCameraConfig = lerpCameraConfig(currentCameraConfig, targetCameraConfig, config.smoothingFactor);
        currentFOV += (targetFOV - currentFOV) * config.smoothingFactor;
        
        // Apply direction multiplier to z (for viewing from opposite side)
        activeCamera.position.set(
          currentCameraConfig.position.x,
          currentCameraConfig.position.y,
          currentCameraConfig.position.z * config.direction
        );
        
        activeCamera.lookAt(
          currentCameraConfig.lookAt.x,
          currentCameraConfig.lookAt.y,
          currentCameraConfig.lookAt.z * config.direction
        );
        
        activeCamera.fov = currentFOV;
        activeCamera.updateProjectionMatrix();
        
        // Update context lookAt target if needed
        if (context.setLookAtTarget) {
          context.setLookAtTarget(
            currentCameraConfig.lookAt.x,
            currentCameraConfig.lookAt.y,
            currentCameraConfig.lookAt.z * config.direction
          );
        }
      }
      
      // Skip if dimensions and zoom haven't changed
      if (currentWidth === lastWidth && 
          currentHeight === lastHeight && 
          config.zoom === lastZoom) {
        return;
      }
      
      lastWidth = currentWidth;
      lastHeight = currentHeight;
      lastZoom = config.zoom;
      
      const aspectRatio = currentWidth / currentHeight;
      
      // Calculate target configuration
      const calculated = calculateCameraConfig(aspectRatio, config.zoom, config.viewedWidth);
      targetCameraConfig = {
        position: calculated.position,
        lookAt: calculated.lookAt
      };
      targetFOV = Math.max(config.minFOV, Math.min(config.maxFOV, calculated.fov));
      
      // Initialize current on first run
      if (!currentCameraConfig) {
        currentCameraConfig = { ...targetCameraConfig };
        currentFOV = targetFOV;
      }
      
      // if (config.debug) {
      //   const screenType = aspectRatio >= ASPECT_RATIO_THRESHOLD ? 'widescreen' : 'portrait';
      //   console.log(`[AspectRatioCamera] Aspect: ${aspectRatio.toFixed(2)} (${screenType})`, {
      //     targetPos: `(${targetCameraConfig.position.x}, ${targetCameraConfig.position.y.toFixed(1)}, ${targetCameraConfig.position.z.toFixed(1)})`,
      //     targetLookAt: `(${targetCameraConfig.lookAt.x}, ${targetCameraConfig.lookAt.y.toFixed(1)}, ${targetCameraConfig.lookAt.z.toFixed(1)})`,
      //     fov: targetFOV.toFixed(1),
      //     zoom: config.zoom.toFixed(2),
      //     direction: config.direction
      //   });
      // }
    }
    
    /**
     * Check for camera readiness
     */
    function checkAndApply() {
      if (context.getCamera?.()) {
        applyCameraPosition();
        return true;
      }
      return false;
    }
    
    // Try immediate application
    if (!checkAndApply()) {
      const readinessInterval = setInterval(() => {
        if (checkAndApply()) {
          clearInterval(readinessInterval);
          if (config.debug) console.log('[AspectRatioCamera] Camera ready, positioning applied');
        }
      }, 200);
      
      setTimeout(() => clearInterval(readinessInterval), 5000);
    }
    
    // Register with context for real-time updates
    context.registerUpdatable?.(applyCameraPosition);
    
    // if (config.debug) {
    //   console.log('[AspectRatioCamera] Initialized with config:', {
    //     basePositions: BASE_POSITIONS,
    //     smoothingFactor: config.smoothingFactor,
    //     zoom: config.zoom,
    //     viewedWidth: config.viewedWidth,
    //     direction: config.direction
    //   });
    // }
    
    // Cleanup function
    return () => {
      // if (config.debug) console.log('[AspectRatioCamera] Cleaning up');
      context.unregisterUpdatable?.(applyCameraPosition);
    };
  };
}