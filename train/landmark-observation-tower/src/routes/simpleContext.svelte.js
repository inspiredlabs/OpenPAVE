// ./simpleContext.svelte.js

/**
 * The Central Nervous System
 * Understanding this file will help you comprehend the whole project's architecture.
 *
 * It is not just a simple data store; it's a living API for each game-world, providing:
 *
 * 1. Single-source-of-truth: Holds references to the global ThreeJS scene (which itself, should be a songleton), renderer, and camera.
 *
 * 2. Shared Data Bus: Stores real-time data from input contollers: poseLandmarks, handLandmarks etc.
 * Any component can access this data without needing it to be passed down through props.
 *
 * 3. The Game Loop Interface: The `updatableFunctions` Set and the registerUpdatable/unregisterUpdatable methods are the genius of this system.
 *
 * They allow any component to "subscribe" its own per-frame logic to the main render loop.
 * 
 */

import { getContext, setContext } from 'svelte';
import * as THREE from 'three';

const CONTEXT_KEY = Symbol('context'); // Updated symbol for clarity

export function createContext() {
  // Internal state for the context
  const state = {
		debugUI: false,
		updaters: new Set(), // The manual subscription system

    // AVBD Physics World (NEW)
    physicsUpdaters: new Set(),
    avbdPhysics: null,
    avbdInitialized: false,

    scene: null,
    renderer: null,
    composer: null, // Add pixelatedRendering.js
    teleNoiseControls: null, // Add teleNoise.js

    // --- Pose-lite or Hands-lite: ---
    modelType: 'hands', // Default
    handLandmarks: null,
    latestHandFrame: null,
    handFrameSequence: 0,
    handFrameSubscribers: new Set(),
    poseLandmarks: null,

    // --- Camera Management ---
    fpsCamera: null,
    droneCamera: null,
    activeCamera: null,       // This will point to either fpsCamera or droneCamera
    activeCameraType: 'fps',  // Default to 'fps'

    // --- NEW PROPERTY FOR BODY MOVEMENT ---
    implicitMoveInput: { x: 0, z: 0 },

    gizmoPosition: { x: 0, y: 0.5, z: 0 },
    canvasWidth: typeof window !== 'undefined' ? window.innerWidth : 1024,
    canvasHeight: typeof window !== 'undefined' ? window.innerHeight : 768,
    updatableFunctions: new Set(),
    gizmoCreated: false,
    lookAtTarget: { x: 0, y: 0, z: 0 }, // For initial setup of FPS camera mainly

    cameraYaw: 0,   // Shared for FPS-style yaw
    cameraPitch: 0, // Shared for FPS-style pitch
    headOrientationControlEnabled: true,
    headOrientationTargetYaw: 0,
    headOrientationTargetPitch: 0,
    hasHeadOrientationTarget: false,
    headOrientationMaxAngularSpeed: Math.PI,
    perfSamples: {
      mainFrames: [],
      mediaPipe: []
    },

    collidableColumnSpheres: new Set(),

    // Duck Hunt game state
    duckHuntState: {
      score: 0,
      lives: 3,
      bullets: 3,
      wave: 1,
      gameState: 'waiting',
      ducksShot: 0,
      ducksToShoot: 2
    },

    // Duck Hunt callback functions (set by components)
    duckHuntStateCallback: null,
    duckHuntStartCallback: null,
    duckHuntResetCallback: null,
    duckHuntGameInstance: null,

    // --- Track objects & add live trails --- (FIXED: Now uses THREE after import)
    trackedPositions: {
        left: new THREE.Vector3(),
        right: new THREE.Vector3()
    },
    isTrackedPositionSet: {
        left: false,
        right: false
    }
  };
  
  const context = {
		get debugUI() { return state.debugUI; },
		set debugUI(value) { 
				state.debugUI = value;
				state.updaters.forEach(fn => fn());
		},
		subscribe: (fn) => state.updaters.add(fn),
		unsubscribe: (fn) => state.updaters.delete(fn),

    // ============================================
    // AVBD PHYSICS METHODS (NEW)
    // ============================================
    // ✅ PHYSICS NOTIFICATION METHODS
    subscribeToPhysics: (fn) => {
      console.log('[Context] Adding physics subscriber, total:', state.physicsUpdaters.size + 1);
      state.physicsUpdaters.add(fn);
    },
    
    unsubscribeFromPhysics: (fn) => {
      console.log('[Context] Removing physics subscriber');
      state.physicsUpdaters.delete(fn);
    },
    
    // ✅ PHYSICS METHODS
    getAVBDPhysics: () => state.avbdPhysics,
    
    isAVBDInitialized: () => state.avbdInitialized,
    
    initializeAVBDPhysics: (physicsInstance) => {
      console.log('[Context] initializeAVBDPhysics called');
      
      if (!physicsInstance) {
        console.warn('[Context] Cannot initialize null physics instance');
        return;
      }
      
      state.avbdPhysics = physicsInstance;
      state.avbdInitialized = true;
      
      // ✅ CRITICAL: Notify ALL subscribers
      console.log('[Context] 🔔 Notifying', state.physicsUpdaters.size, 'physics subscribers');
      state.physicsUpdaters.forEach(fn => {
        console.log('[Context] Calling subscriber...');
        try {
          fn();
        } catch (error) {
          console.error('[Context] Subscriber error:', error);
        }
      });
      
      // Register physics step
      const physicsStepFn = async (deltaTime, camera) => {
        if (physicsInstance.initialized && !physicsInstance.stepping) {
          await physicsInstance.step();
        }
      };
      
      context.registerUpdatable(physicsStepFn);
      
      console.log('[Context] ✅ AVBD physics fully registered');
    },

    // Swap through AVBD features
    clearPhysics: () => {
      if (state.avbdPhysics) {
        state.avbdPhysics.clear();
        console.log('[Context] Physics cleared');
      }
    },
    
    destroyAVBDPhysics: () => {
      if (state.avbdPhysics) {
        state.avbdPhysics.destroy();
        state.avbdPhysics = null;
        state.avbdInitialized = false;
        console.log('[Context] AVBD physics destroyed');
      }
    },


    // Scene
    getScene: () => state.scene,
    setScene: (newScene) => { state.scene = newScene; },

    // Renderer
    getRenderer: () => state.renderer,
    setRenderer: (newRenderer) => { state.renderer = newRenderer; },

    // Add pixelatedRendering.js
    getComposer: () => state.composer,
    setComposer: (newComposer) => { state.composer = newComposer; },

    // Add teleNoise.js
    getTeleNoiseControls: () => state.teleNoiseControls,
    setTeleNoiseControls: (controls) => { state.teleNoiseControls = controls; },

    // Pose-lite getter and setter for pose landmarks
    getPoseLandmarks: () => state.poseLandmarks,
    setPoseLandmarks: (landmarks) => { state.poseLandmarks = landmarks; },
    setModelType: (type) => { state.modelType = type; },
    getModelType: () => state.modelType,
    setHandLandmarks: (landmarks) => { state.handLandmarks = landmarks; },
    getHandLandmarks: () => state.handLandmarks,
    publishHandFrame: (frame) => {
      if (!frame || !Number.isFinite(frame.capturedAt)) return false;
      if (state.latestHandFrame && frame.capturedAt <= state.latestHandFrame.capturedAt) return false;

      const publishedFrame = {
        ...frame,
        sequence: ++state.handFrameSequence
      };

      // Bounded latest-value channel: the previous frame becomes collectible
      // immediately and no history is retained in the shared context.
      state.latestHandFrame = publishedFrame;
      for (const subscriber of state.handFrameSubscribers) {
        try {
          subscriber(publishedFrame);
        } catch (error) {
          console.error('[Context] Hand frame subscriber failed:', error);
        }
      }
      return publishedFrame;
    },
    getLatestHandFrame: () => state.latestHandFrame,
    subscribeHandFrames: (subscriber) => {
      if (typeof subscriber !== 'function') return () => {};
      state.handFrameSubscribers.add(subscriber);
      if (state.latestHandFrame) {
        try {
          subscriber(state.latestHandFrame);
        } catch (error) {
          console.error('[Context] Hand frame subscriber failed:', error);
        }
      }
      return () => state.handFrameSubscribers.delete(subscriber);
    },
    clearHandFrame: () => {
      state.latestHandFrame = null;
      for (const subscriber of state.handFrameSubscribers) {
        try {
          subscriber(null);
        } catch (error) {
          console.error('[Context] Hand frame subscriber failed:', error);
        }
      }
    },

    // --- Camera Management Methods ---
    setFpsCamera: (cam) => {
      state.fpsCamera = cam;
      // If FPS is the intended active type OR no active camera is set yet, make this active.
      if (state.activeCameraType === 'fps' || !state.activeCamera) {
        state.activeCamera = state.fpsCamera;
        state.activeCameraType = 'fps'; // Ensure type is correctly set
      }
    },
    getFpsCamera: () => state.fpsCamera,

    setDroneCamera: (cam) => {
      state.droneCamera = cam;
      // If drone is the intended active type, make this active.
      if (state.activeCameraType === 'drone') {
        state.activeCamera = state.droneCamera;
      }
    },
    getDroneCamera: () => state.droneCamera,

    setActiveCameraType: (type) => { // type: 'fps' or 'drone'
      if (type === 'fps' && state.fpsCamera) {
        state.activeCamera = state.fpsCamera;
        state.activeCameraType = 'fps';
      } else if (type === 'drone' && state.droneCamera) {
        state.activeCamera = state.droneCamera;
        state.activeCameraType = 'drone';
      } else {
        console.warn(`[Context] Attempted to switch to unknown or unset camera type: ${type}`);
      }
    },
    getActiveCameraType: () => state.activeCameraType,

    // Generic camera getter - returns the currently active camera
    getCamera: () => state.activeCamera,

    // --- NEW METHODS FOR BODY MOVEMENT ---
    getImplicitMoveInput: () => state.implicitMoveInput,
    setImplicitMoveInput: (input) => {
        if (input) {
            state.implicitMoveInput = input;
        }
    },

    getGizmoPosition: () => ({ ...state.gizmoPosition }),
    updateGizmoAxis: (axis, value) => {
      if (axis in state.gizmoPosition) {
        state.gizmoPosition[axis] = value;
      }
    },

    setLookAtTarget: (x, y, z) => { state.lookAtTarget = { x, y, z }; },
    getLookAtTarget: () => ({ ...state.lookAtTarget }),

    get canvasWidth() { return state.canvasWidth; },
    set canvasWidth(value) { state.canvasWidth = value; },
    get canvasHeight() { return state.canvasHeight; },
    set canvasHeight(value) { state.canvasHeight = value; },

    // Direct access properties
    get camera() { return state.activeCamera; },
    get renderer() { return state.renderer; },
    set renderer(r) { state.renderer = r; },

    // Status getters
    get threeJsScene() { return state.scene; },
    get threeJsCamera() { return state.activeCamera; },
    get threeJsReady() { return state.scene !== null && state.activeCamera !== null; },

    // FPS Camera Orientation (Yaw and Pitch)
    getCameraYaw: () => state.cameraYaw,
    setCameraYaw: (value) => { state.cameraYaw = value; },
    getCameraPitch: () => state.cameraPitch,
    setCameraPitch: (value) => {
      state.cameraPitch = Math.max(-Math.PI / 2 + 0.001, Math.min(Math.PI / 2 - 0.001, value));
    },
    setHeadOrientationTarget: (yaw, pitch) => {
      state.headOrientationTargetYaw = yaw;
      state.headOrientationTargetPitch = Math.max(-Math.PI / 2 + 0.001, Math.min(Math.PI / 2 - 0.001, pitch));
      state.hasHeadOrientationTarget = true;
    },
    clearHeadOrientationTarget: () => {
      state.hasHeadOrientationTarget = false;
    },
    getHeadOrientationMaxAngularSpeed: () => state.headOrientationMaxAngularSpeed,
    setHeadOrientationMaxAngularSpeed: (radiansPerSecond) => {
      if (Number.isFinite(radiansPerSecond) && radiansPerSecond > 0) {
        state.headOrientationMaxAngularSpeed = radiansPerSecond;
      }
    },
    applyHeadOrientationSmoothing: (deltaTime = 1 / 60) => {
      if (!state.headOrientationControlEnabled || !state.hasHeadOrientationTarget) return;

      const frameDelta = Math.min(Math.max(deltaTime, 0), 1 / 30);
      const maxStep = state.headOrientationMaxAngularSpeed * frameDelta;
      const moveTowards = (current, target) => {
        const delta = target - current;
        if (Math.abs(delta) <= maxStep) return target;
        return current + Math.sign(delta) * maxStep;
      };

      state.cameraYaw = moveTowards(state.cameraYaw, state.headOrientationTargetYaw);
      state.cameraPitch = moveTowards(state.cameraPitch, state.headOrientationTargetPitch);
      state.cameraPitch = Math.max(-Math.PI / 2 + 0.001, Math.min(Math.PI / 2 - 0.001, state.cameraPitch));
    },
    getHeadOrientationControlEnabled: () => state.headOrientationControlEnabled,
    setHeadOrientationControlEnabled: (enabled) => {
      state.headOrientationControlEnabled = !!enabled;
      if (!state.headOrientationControlEnabled) {
        state.hasHeadOrientationTarget = false;
      }
      context.mediaPipeApi?.updateConfig?.({
        enableHeadOrientationControl: state.headOrientationControlEnabled
      });
    },
    recordMainFramePerf: (sample) => {
      state.perfSamples.mainFrames.push({ t: performance.now(), ...sample });
      if (state.perfSamples.mainFrames.length > 600) state.perfSamples.mainFrames.shift();
    },
    recordMediaPipePerf: (sample) => {
      state.perfSamples.mediaPipe.push({ t: performance.now(), ...sample });
      if (state.perfSamples.mediaPipe.length > 600) state.perfSamples.mediaPipe.shift();
    },
    mediapipePerfTrace: ({ durationMs = 10_000, log = true } = {}) => {
      const now = performance.now();
      const since = now - durationMs;
      const mainFrames = state.perfSamples.mainFrames.filter((sample) => sample.t >= since);
      const mediaPipe = state.perfSamples.mediaPipe.filter((sample) => sample.t >= since);

      const summarize = (samples, key) => {
        const values = samples
          .map((sample) => sample[key])
          .filter((value) => Number.isFinite(value))
          .sort((a, b) => a - b);

        if (!values.length) return null;
        const sum = values.reduce((total, value) => total + value, 0);
        const percentile = (p) => values[Math.min(values.length - 1, Math.floor((values.length - 1) * p))];

        return {
          avg: +(sum / values.length).toFixed(2),
          p50: +percentile(0.5).toFixed(2),
          p95: +percentile(0.95).toFixed(2),
          max: +values[values.length - 1].toFixed(2)
        };
      };

      const summary = {
        windowMs: durationMs,
        capturedAt: new Date().toISOString(),
        main: {
          frames: mainFrames.length,
          frameMs: summarize(mainFrames, 'frameMs'),
          updatersMs: summarize(mainFrames, 'updatersMs'),
          renderMs: summarize(mainFrames, 'renderMs')
        },
        mediaPipe: {
          frames: mediaPipe.length,
          totalMs: summarize(mediaPipe, 'totalMs'),
          inputCanvasMs: summarize(mediaPipe, 'inputCanvasMs'),
          poseMs: summarize(mediaPipe, 'poseMs'),
          handMs: summarize(mediaPipe, 'handMs'),
          drawMs: summarize(mediaPipe, 'drawMs'),
          updateMs: summarize(mediaPipe, 'updateMs'),
          ranPose: mediaPipe.filter((sample) => sample.ranPose).length,
          ranHands: mediaPipe.filter((sample) => sample.ranHands).length,
          skippedPoseDelta: mediaPipe.filter((sample) => sample.skippedPoseDelta).length,
          latestConfig: mediaPipe.at(-1)?.config ?? context.mediaPipeApi?.getConfig?.()
        }
      };

      if (log) {
        console.log('[MP Perf Trace]', summary);
        if (mediaPipe.length) console.table(mediaPipe.slice(-20));
      }

      return summary;
    },

    // Updatable functions
    registerUpdatable: (fn) => { if (typeof fn === 'function') state.updatableFunctions.add(fn); },
    unregisterUpdatable: (fn) => { state.updatableFunctions.delete(fn); },
    _getUpdatableFunctions: () => state.updatableFunctions,

    // Collidable Spheres (EXISTING - don't duplicate)
    registerCollidableSphere: (sphereData) => { if (sphereData && sphereData.id) state.collidableColumnSpheres.add(sphereData); },
    unregisterCollidableSphere: (sphereData) => { state.collidableColumnSpheres.delete(sphereData); },
    unregisterCollidableSpheresByParentId: (parentId) => {
        const spheresToRemove = [];
        for (const sphere of state.collidableColumnSpheres) {
            if (sphere.parentId === parentId) spheresToRemove.push(sphere);
        }
        spheresToRemove.forEach(sphere => state.collidableColumnSpheres.delete(sphere));
    },
    getCollidableSpheres: () => Array.from(state.collidableColumnSpheres),

    // Duck Hunt state management
    getDuckHuntState: () => ({ ...state.duckHuntState }),
    updateDuckHuntState: (newState) => {
      state.duckHuntState = { ...state.duckHuntState, ...newState };
      // Call registered callback if it exists
      if (state.duckHuntStateCallback) {
        state.duckHuntStateCallback(state.duckHuntState);
      }
    },

    // Duck Hunt callback registration
    setDuckHuntState: (callback) => {
      state.duckHuntStateCallback = callback;
    },
    duckHuntStart: (callback) => {
      if (callback) {
        state.duckHuntStartCallback = callback;
      } else if (state.duckHuntStartCallback) {
        state.duckHuntStartCallback();
      }
    },
    duckHuntReset: (callback) => {
      if (callback) {
        state.duckHuntResetCallback = callback;
      } else if (state.duckHuntResetCallback) {
        state.duckHuntResetCallback();
      }
    },

    // Duck Hunt game instance management
    get duckHuntGameInstance() { return state.duckHuntGameInstance; },
    set duckHuntGameInstance(instance) { state.duckHuntGameInstance = instance; },

    // --- Track objects & add live trails --- (NEW METHODS)
    setTrackedPosition: (hand, position) => {
        if (state.trackedPositions[hand] && position) {
            state.trackedPositions[hand].copy(position);
            state.isTrackedPositionSet[hand] = true;
        } else {
            state.isTrackedPositionSet[hand] = false;
        }
    },
    getTrackedPosition: (hand) => {
        // Return the position only if it has been set, otherwise return null
        return state.isTrackedPositionSet[hand] ? state.trackedPositions[hand] : null;
    },

    // Helper to check if any tracked positions are set
    hasTrackedPositions: () => {
        return state.isTrackedPositionSet.left || state.isTrackedPositionSet.right;
    },

    // Reset tracked positions
    resetTrackedPositions: () => {
        state.isTrackedPositionSet.left = false;
        state.isTrackedPositionSet.right = false;
        state.trackedPositions.left.set(0, 0, 0);
        state.trackedPositions.right.set(0, 0, 0);
    }
  };

  setContext(CONTEXT_KEY, context);
  return context;
}

// Consume the context
export function useContext() {
  return getContext(CONTEXT_KEY);
}
