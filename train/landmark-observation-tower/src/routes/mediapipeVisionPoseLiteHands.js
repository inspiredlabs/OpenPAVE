// @ts-nocheck
import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker
} from '@mediapipe/tasks-vision';
import { useContext } from './simpleContext.svelte.js';
import { buildOracleHand, ORACLE_FRAME_SCHEMA } from './handOracleGeometry.js';

export function mediapipeVisionPoseLiteHands(options = {}) {
  return () => {
    console.log("Setting up MediaPipe Pose Lite + Hands attachment");

    const context = useContext();
    const isIOS =
      /iPad|iPhone|iPod/.test(navigator.userAgent) ||
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

    let poseLandmarker = null;
    let handLandmarker = null;
    let video = null;
    let previewVideo = null;
    let poseCanvas = null;
    let handCanvas = null;
    let processCanvas = null;
    let processCtx = null;
    let deltaCanvas = null;
    let deltaCtx = null;
    let poseDrawingUtils = null;
    let handDrawingUtils = null;
    let lastVideoTime = -1;
    let isInitialized = false;

    const skeletonsEnabled = options.drawSkeletons ?? true;
    const config = {
      videoOpacity: options.videoOpacity ?? 0.0,
      previewOpacity: options.previewOpacity ?? 1.0,
      drawPoseOverlay: options.drawPoseOverlay ?? skeletonsEnabled,
      drawHandPreview: options.drawHandPreview ?? skeletonsEnabled,
      filterAlpha: options.filterAlpha ?? 0.6,
      handFilterAlpha: options.handFilterAlpha ?? options.filterAlpha ?? 0.8,
      viewportScale: options.viewportScale ?? 0.2,
      enableHeadOrientationControl: options.enableHeadOrientationControl ?? true,
      maxYawDegrees: options.maxYawDegrees ?? 5,
      maxPitchDegrees: options.maxPitchDegrees ?? 60,
      orientationFilterAlpha: options.orientationFilterAlpha ?? 0.4,
      enablePoseDetection: options.enablePoseDetection ?? true,
      enableHandDetection: options.enableHandDetection ?? true,
      maxHands: options.maxHands ?? 1,
      optimized: options.optimized ?? true,
      processWidth: options.processWidth ?? 384,
      processHeight: options.processHeight ?? null,
      poseTargetFps: options.poseTargetFps ?? 12,
      handTargetFps: options.handTargetFps ?? 30,
      handTelemetryTargetFps: options.handTelemetryTargetFps ?? 10,
      enableSceneDeltaSkip: options.enableSceneDeltaSkip ?? true,
      sceneDeltaThreshold: options.sceneDeltaThreshold ?? 3.0,
      forcePoseRefreshMs: options.forcePoseRefreshMs ?? 500,
      workerIsolation: false,
      minHandDetectionConfidence: options.minHandDetectionConfidence ?? 0.5,
      minHandPresenceConfidence: options.minHandPresenceConfidence ?? 0.5,
      delegate: options.delegate ?? (isIOS ? 'CPU' : 'GPU'),
      ...options
    };

    let smoothedPoseLandmarks = null;
    let smoothedPrimaryHandLandmarks = null;
    let prevHeadPosition = { x: 0, y: 0, z: 0 };
    let prevCalculatedYaw = 0;
    let prevCalculatedPitch = 0;
    let wasHeadOrientationEnabled = true;
    let lastPoseCheckTime = 0;
    let lastPoseRunTime = 0;
    let lastHandRunTime = 0;
    let lastHasPose = false;
    let lastHasHands = false;
    let previousDeltaPixels = null;
    let lastHandTelemetryTime = -Infinity;
    let lastPublishedHandCount = 0;

    function applyLowPassFilter(newValue, prevValue, alpha) {
      if (prevValue === undefined || prevValue === null) return newValue;
      if (alpha >= 0.99) return newValue;
      const mappedAlpha = alpha * alpha;
      return mappedAlpha * newValue + (1 - mappedAlpha) * prevValue;
    }

    function mapLandmarkToWorld(landmark) {
      if (!landmark) return { x: 0, y: 0, z: 0 };
      return {
        x: (landmark.x * 2 - 1) * config.viewportScale,
        y: -(landmark.y * 2 - 1) * config.viewportScale,
        z: landmark.z * config.viewportScale
      };
    }

    function clearCanvas(canvasElement) {
      if (!canvasElement) return;
      const ctx = canvasElement.getContext('2d');
      ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }

    function updatePreviewVisibility() {
      if (!previewVideo || !handCanvas) return;
      const handPreviewVisible = config.drawHandPreview !== false;
      previewVideo.style.display = handPreviewVisible ? 'block' : 'none';
      handCanvas.style.display = handPreviewVisible ? 'block' : 'none';
      previewVideo.style.opacity = String(config.previewOpacity);
      handCanvas.style.opacity = String(config.previewOpacity);
      if (!handPreviewVisible) {
        clearCanvas(handCanvas);
      }
    }

    function configureProcessingCanvases() {
      if (!video?.videoWidth || !video?.videoHeight) return;

      const aspect = video.videoHeight / video.videoWidth;
      const width = Math.max(160, Math.floor(config.processWidth));
      const height = Math.max(90, Math.floor(config.processHeight ?? width * aspect));

      processCanvas.width = width;
      processCanvas.height = height;
      deltaCanvas.width = 32;
      deltaCanvas.height = Math.max(18, Math.round(32 * aspect));
      previousDeltaPixels = null;
    }

    function updateProcessingInput() {
      if (!config.optimized || !processCanvas || !processCtx) return video;
      processCtx.drawImage(video, 0, 0, processCanvas.width, processCanvas.height);
      return processCanvas;
    }

    function measureSceneDelta(source) {
      if (!config.enableSceneDeltaSkip || !deltaCanvas || !deltaCtx) {
        return { changed: true, delta: null };
      }

      deltaCtx.drawImage(source || video, 0, 0, deltaCanvas.width, deltaCanvas.height);
      const pixels = deltaCtx.getImageData(0, 0, deltaCanvas.width, deltaCanvas.height).data;

      if (!previousDeltaPixels) {
        previousDeltaPixels = new Uint8ClampedArray(pixels);
        return { changed: true, delta: null };
      }

      let total = 0;
      const sampleCount = pixels.length / 4;
      for (let i = 0; i < pixels.length; i += 4) {
        const currentLuma = 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
        const previousLuma =
          0.2126 * previousDeltaPixels[i] +
          0.7152 * previousDeltaPixels[i + 1] +
          0.0722 * previousDeltaPixels[i + 2];
        total += Math.abs(currentLuma - previousLuma);
      }

      previousDeltaPixels.set(pixels);
      const delta = total / sampleCount;
      return {
        changed: delta >= config.sceneDeltaThreshold,
        delta
      };
    }

    function updateHeadOrientationControl(nose) {
      if (!nose) return;

      const rawPosition = mapLandmarkToWorld(nose);
      const smoothedPosition = {
        x: applyLowPassFilter(rawPosition.x, prevHeadPosition.x, config.filterAlpha),
        y: applyLowPassFilter(rawPosition.y, prevHeadPosition.y, config.filterAlpha),
        z: applyLowPassFilter(rawPosition.z, prevHeadPosition.z, config.filterAlpha)
      };

      context.headPosition = { ...smoothedPosition };
      prevHeadPosition = { ...smoothedPosition };

      const headOrientationEnabled =
        config.enableHeadOrientationControl &&
        (context.getHeadOrientationControlEnabled?.() ?? true);
      const horizontalDeviation = nose.x - 0.5;
      const verticalDeviation = nose.y - 0.5;
      const targetYaw = -horizontalDeviation * 2 * (config.maxYawDegrees * Math.PI / 180);
      const targetPitch = -verticalDeviation * 2 * (config.maxPitchDegrees * Math.PI / 180);

      if (!headOrientationEnabled) {
        wasHeadOrientationEnabled = false;
        context.clearHeadOrientationTarget?.();
        return;
      }

      if (!wasHeadOrientationEnabled) {
        wasHeadOrientationEnabled = true;
        prevCalculatedYaw = context.getCameraYaw?.() ?? targetYaw;
        prevCalculatedPitch = context.getCameraPitch?.() ?? targetPitch;
      }

      if (context.setHeadOrientationTarget) {
        const smoothedYaw = applyLowPassFilter(targetYaw, prevCalculatedYaw, config.orientationFilterAlpha);
        const smoothedPitch = applyLowPassFilter(targetPitch, prevCalculatedPitch, config.orientationFilterAlpha);

        context.setHeadOrientationTarget(smoothedYaw, smoothedPitch);
        prevCalculatedYaw = smoothedYaw;
        prevCalculatedPitch = smoothedPitch;
      }
    }

    function updatePoseTracking(poseResults) {
      if (!config.enablePoseDetection || !poseResults?.landmarks?.length) {
        context.setPoseLandmarks?.(null);
        smoothedPoseLandmarks = null;
        return false;
      }

      const rawPoseLandmarks = poseResults.landmarks[0];

      if (!smoothedPoseLandmarks) {
        smoothedPoseLandmarks = JSON.parse(JSON.stringify(rawPoseLandmarks));
      } else {
        for (let i = 0; i < rawPoseLandmarks.length; i++) {
          const previous = smoothedPoseLandmarks[i];
          const current = rawPoseLandmarks[i];
          smoothedPoseLandmarks[i] = {
            x: applyLowPassFilter(current.x, previous.x, config.filterAlpha),
            y: applyLowPassFilter(current.y, previous.y, config.filterAlpha),
            z: applyLowPassFilter(current.z, previous.z, config.filterAlpha),
            visibility: current.visibility
          };
        }
      }

      context.setPoseLandmarks?.(smoothedPoseLandmarks);
      updateHeadOrientationControl(smoothedPoseLandmarks[0]);
      return true;
    }

    function captureHandFrame(handResults, capturedAt) {
      const sourceLandmarks = handResults?.landmarks ?? [];
      const sourceWorldLandmarks = handResults?.worldLandmarks ?? [];
      const sourceHandedness = handResults?.handedness ?? [];

      const handedness = sourceHandedness.map((categories) =>
        categories.map(({ index, score, categoryName, displayName }) => ({
          index,
          score,
          categoryName,
          displayName
        }))
      );
      const landmarks = sourceLandmarks.map((hand) =>
        hand.map(({ x, y, z, visibility, presence }) => ({
          x,
          y,
          z,
          visibility: Number.isFinite(visibility) ? visibility : null,
          presence: Number.isFinite(presence) ? presence : null
        }))
      );
      const worldLandmarks = sourceWorldLandmarks.map((hand) =>
        hand.map(({ x, y, z, visibility, presence }) => ({
          x,
          y,
          z,
          visibility: Number.isFinite(visibility) ? visibility : null,
          presence: Number.isFinite(presence) ? presence : null
        }))
      );

      // All result fields are copied synchronously before MediaPipe can reuse
      // its result buffers. This object is published once and never mutated.
      return {
        schema: ORACLE_FRAME_SCHEMA,
        capturedAt,
        timestampUnixMs: performance.timeOrigin + capturedAt,
        handCount: landmarks.length,
        source: {
          videoWidth: video?.videoWidth ?? null,
          videoHeight: video?.videoHeight ?? null,
          inferenceWidth: config.optimized ? processCanvas?.width ?? null : video?.videoWidth ?? null,
          inferenceHeight: config.optimized ? processCanvas?.height ?? null : video?.videoHeight ?? null
        },
        camera: {
          facingMode: 'user',
          inferenceMirrored: false,
          displayMirrored: true
        },
        coordinateContract: {
          normalizedImage: {
            x: 'right',
            y: 'down',
            z: 'wrist_relative_2.5d_teacher_depth',
            range: 'x_y_nominally_0_to_1'
          },
          world: {
            unit: 'metre',
            origin: 'mediapipe_hand_geometric_centre',
            axes: 'mediapipe_tasks_raw_unmodified'
          },
          normals: {
            required: ['palm_plane', 'bone_frames', 'joint_bending_planes'],
            unavailable: ['skin_surface_normals'],
            supervision: 'mediapipe_world_landmark_pseudo_label'
          }
        },
        handedness,
        landmarks,
        worldLandmarks,
        oracleHands: landmarks.map((hand, handIndex) =>
          buildOracleHand(hand, worldLandmarks[handIndex] ?? [], handedness[handIndex] ?? [], handIndex)
        )
      };
    }

    function publishHandTelemetry(handResults, currentTime) {
      const handCount = handResults?.landmarks?.length ?? 0;
      const intervalMs = 1000 / Math.max(1, config.handTelemetryTargetFps);
      const handPresenceChanged = handCount !== lastPublishedHandCount;

      // One empty frame is enough to represent the hand-lost transition.
      if (handCount === 0 && lastPublishedHandCount === 0) return;
      if (!handPresenceChanged && currentTime - lastHandTelemetryTime < intervalMs) return;

      lastHandTelemetryTime = currentTime;
      lastPublishedHandCount = handCount;
      context.publishHandFrame?.(captureHandFrame(handResults, currentTime));
    }

    function updateHandTracking(handResults, currentTime) {
      publishHandTelemetry(handResults, currentTime);

      if (!config.enableHandDetection || !handResults?.landmarks?.length) {
        context.setHandLandmarks?.(null);
        smoothedPrimaryHandLandmarks = null;
        if (context.setHandFound) context.setHandFound(false);
        return false;
      }

      const rawHandLandmarks = handResults.landmarks[0];

      if (!smoothedPrimaryHandLandmarks) {
        smoothedPrimaryHandLandmarks = JSON.parse(JSON.stringify(rawHandLandmarks));
      } else {
        for (let i = 0; i < rawHandLandmarks.length; i++) {
          const previous = smoothedPrimaryHandLandmarks[i];
          const current = rawHandLandmarks[i];
          smoothedPrimaryHandLandmarks[i] = {
            x: applyLowPassFilter(current.x, previous.x, config.handFilterAlpha),
            y: applyLowPassFilter(current.y, previous.y, config.handFilterAlpha),
            z: applyLowPassFilter(current.z, previous.z, config.handFilterAlpha),
            visibility: current.visibility
          };
        }
      }

      context.setHandLandmarks?.(smoothedPrimaryHandLandmarks);
      if (context.setHandFound) context.setHandFound(true);
      return true;
    }

    function drawPoseOverlay(poseResults) {
      if (!poseCanvas || !poseDrawingUtils) return;
      const ctx = poseCanvas.getContext('2d');
      ctx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);

      if (!config.drawPoseOverlay || !poseResults?.landmarks) return;

      for (const landmarks of poseResults.landmarks) {
        poseDrawingUtils.drawLandmarks(landmarks, {
          radius: 3,
          color: '#00FF00'
        });
        poseDrawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
          color: '#FF0000',
          lineWidth: 2
        });
      }
    }

    function drawHandPreview(handResults) {
      if (!handCanvas || !handDrawingUtils) return;
      const ctx = handCanvas.getContext('2d');
      ctx.clearRect(0, 0, handCanvas.width, handCanvas.height);

      if (!config.drawHandPreview || !handResults?.landmarks) return;

      for (const landmarks of handResults.landmarks) {
        handDrawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
          color: '#00FF00',
          lineWidth: 2
        });
        handDrawingUtils.drawLandmarks(landmarks, {
          color: '#FF0000',
          radius: 4
        });
      }
    }

    function getDelegateOrder() {
      const normalizedDelegate = String(config.delegate || '').trim().toUpperCase();
      if (normalizedDelegate === 'AUTO') {
        return isIOS ? ['CPU'] : ['GPU', 'CPU'];
      }
      return normalizedDelegate === 'GPU' ? ['GPU', 'CPU'] : ['CPU'];
    }

    async function createPoseLandmarker(vision) {
      let initError = null;
      for (const delegate of getDelegateOrder()) {
        try {
          return await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
              delegate
            },
            runningMode: 'VIDEO',
            numPoses: 1
          });
        } catch (error) {
          initError = error;
          console.warn(`[MediaPipe PoseLite] ${delegate} delegate init failed:`, error);
        }
      }
      throw initError || new Error('Unable to initialize MediaPipe Pose Lite');
    }

    async function createHandLandmarker(vision) {
      let initError = null;
      for (const delegate of getDelegateOrder()) {
        try {
          return await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
              delegate
            },
            runningMode: 'VIDEO',
            numHands: config.maxHands,
            minHandDetectionConfidence: config.minHandDetectionConfidence,
            minHandPresenceConfidence: config.minHandPresenceConfidence,
            minTrackingConfidence: 0.5
          });
        } catch (error) {
          initError = error;
          console.warn(`[MediaPipe Hands] ${delegate} delegate init failed:`, error);
        }
      }
      throw initError || new Error('Unable to initialize MediaPipe Hands');
    }

    async function initializeMediaPipe() {
      if (isInitialized) return true;

      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
        );

        if (config.enablePoseDetection) {
          poseLandmarker = await createPoseLandmarker(vision);
          console.log("MediaPipe Pose Lite model loaded successfully");
        }
        if (config.enableHandDetection) {
          handLandmarker = await createHandLandmarker(vision);
          console.log("MediaPipe Hands model loaded successfully");
        }

        poseDrawingUtils = new DrawingUtils(poseCanvas.getContext('2d'));
        handDrawingUtils = new DrawingUtils(handCanvas.getContext('2d'));
        isInitialized = true;
        context.mediaPipeInitialized = true;
        return true;
      } catch (error) {
        console.error('Failed to initialize pose-lite + hands:', error);
        context.mediaPipeInitialized = false;
        context.errorMessage = `MediaPipe error: ${error.message}`;
        return false;
      }
    }

    function setupVideoElements() {
      video = document.createElement('video');
      video.autoplay = true;
      video.playsInline = true;
      video.muted = true;
      video.className = 'selfie';
      video.style.cssText = `
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        object-fit: cover;
        z-index: 1;
        opacity: ${config.videoOpacity};
        transform: scaleX(-1);
        pointer-events: none;
      `;
      document.body.appendChild(video);

      previewVideo = document.createElement('video');
      previewVideo.autoplay = true;
      previewVideo.playsInline = true;
      previewVideo.muted = true;
      previewVideo.className = 'selfie';
      previewVideo.style.cssText = `
        position: fixed; bottom: 20px; left: calc(50vw - 2.5rem);
        width: 5rem; height: 10rem; object-fit: cover;
        opacity: ${config.previewOpacity};
        transform: scaleX(-1);
        z-index: 4;
        border: 0.15rem solid #383838;
        border-radius: 25% / 12.5%;
        box-shadow: 0 0.125rem 0.2rem rgba(0,0,0, 1);
        pointer-events: none;
      `;
      document.body.appendChild(previewVideo);

      poseCanvas = document.createElement('canvas');
      poseCanvas.className = 'pose-canvas selfie';
      poseCanvas.style.cssText = `
        position: absolute;
        z-index: 2;
        pointer-events: none;
        top: 0; left: 0;
        width: 100%; height: 100%;
        object-fit: cover;
        transform: scaleX(-1);
      `;
      document.body.appendChild(poseCanvas);

      handCanvas = document.createElement('canvas');
      handCanvas.className = 'hand-canvas selfie';
      handCanvas.style.cssText = `
        position: fixed; bottom: 20px; left: calc(50vw - 2.5rem);
        width: 5rem; height: 10rem; object-fit: cover;
        opacity: ${config.previewOpacity};
        transform: scaleX(-1);
        z-index: 5;
        border: 0.15rem solid #383838;
        border-radius: 25% / 12.5%;
        box-shadow: 0 0.125rem 0.2rem rgba(0,0,0, 1);
        pointer-events: none;
      `;
      document.body.appendChild(handCanvas);

      processCanvas = document.createElement('canvas');
      processCtx = processCanvas.getContext('2d', { alpha: false });
      deltaCanvas = document.createElement('canvas');
      deltaCtx = deltaCanvas.getContext('2d', { willReadFrequently: true });

      updatePreviewVisibility();
    }

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          }
        });

        video.srcObject = stream;
        previewVideo.srcObject = stream;

        await new Promise((resolve) => {
          video.onloadeddata = () => {
            poseCanvas.width = video.videoWidth;
            poseCanvas.height = video.videoHeight;
            handCanvas.width = video.videoWidth;
            handCanvas.height = video.videoHeight;
            configureProcessingCanvases();
            resolve();
          };
        });

        return true;
      } catch (error) {
        console.error('Failed to access camera:', error);
        context.errorMessage = `Camera error: ${error.message}`;
        return false;
      }
    }

    function processMediaPipeFrame(_deltaTime, currentTime) {
      if (!video || !isInitialized || video.paused || video.ended) return;

      if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
        lastVideoTime = video.currentTime;

        const perfStart = performance.now();
        let inputCanvasMs = 0;
        let poseMs = 0;
        let handMs = 0;
        let updateMs = 0;
        let drawMs = 0;
        let ranPose = false;
        let ranHands = false;
        let skippedPoseDelta = false;
        let sceneDelta = null;
        const poseIntervalMs = 1000 / Math.max(1, config.poseTargetFps);
        const handIntervalMs = 1000 / Math.max(1, config.handTargetFps);
        const poseCheckDue = currentTime - lastPoseCheckTime >= poseIntervalMs;
        const handDue = currentTime - lastHandRunTime >= handIntervalMs;

        if (!poseCheckDue && !handDue) return;

        try {
          const inputStart = performance.now();
          const inferenceSource = updateProcessingInput();
          inputCanvasMs = performance.now() - inputStart;

          let shouldRunPose = poseCheckDue;
          if (poseCheckDue) {
            lastPoseCheckTime = currentTime;
            const deltaResult = measureSceneDelta(inferenceSource);
            sceneDelta = deltaResult.delta;
            shouldRunPose =
              !config.enableSceneDeltaSkip ||
              deltaResult.changed ||
              (config.forcePoseRefreshMs > 0 && currentTime - lastPoseRunTime >= config.forcePoseRefreshMs);
            skippedPoseDelta = !shouldRunPose;
          }

          let poseResults = null;
          let handResults = null;

          if (config.enablePoseDetection && poseLandmarker && shouldRunPose) {
            const poseStart = performance.now();
            poseResults = poseLandmarker.detectForVideo(inferenceSource, currentTime);
            poseMs = performance.now() - poseStart;
            lastPoseRunTime = currentTime;
            ranPose = true;
          }

          if (config.enableHandDetection && handLandmarker && handDue) {
            const handStart = performance.now();
            handResults = handLandmarker.detectForVideo(inferenceSource, currentTime);
            handMs = performance.now() - handStart;
            lastHandRunTime = currentTime;
            ranHands = true;
          }

          const updateStart = performance.now();
          if (ranPose) lastHasPose = updatePoseTracking(poseResults);
          if (ranHands) lastHasHands = updateHandTracking(handResults, currentTime);
          updateMs = performance.now() - updateStart;

          const drawStart = performance.now();
          if (ranPose) drawPoseOverlay(poseResults);
          if (ranHands) drawHandPreview(handResults);
          drawMs = performance.now() - drawStart;

          context.isTracking = lastHasPose || lastHasHands;
          context.recordMediaPipePerf?.({
            totalMs: performance.now() - perfStart,
            inputCanvasMs,
            poseMs,
            handMs,
            updateMs,
            drawMs,
            ranPose,
            ranHands,
            skippedPoseDelta,
            sceneDelta,
            sourceWidth: video.videoWidth,
            sourceHeight: video.videoHeight,
            processWidth: inferenceSource?.width ?? video.videoWidth,
            processHeight: inferenceSource?.height ?? video.videoHeight,
            config: {
              optimized: config.optimized,
              workerIsolation: config.workerIsolation,
              poseTargetFps: config.poseTargetFps,
              handTargetFps: config.handTargetFps,
              processWidth: config.optimized ? processCanvas?.width : video.videoWidth,
              processHeight: config.optimized ? processCanvas?.height : video.videoHeight,
              enableSceneDeltaSkip: config.enableSceneDeltaSkip,
              sceneDeltaThreshold: config.sceneDeltaThreshold,
              forcePoseRefreshMs: config.forcePoseRefreshMs
            }
          });
        } catch (error) {
          console.error("Error processing pose-lite + hand frame:", error);
        }
      }
    }

    async function initialize() {
      setupVideoElements();
      const mediaPipeReady = await initializeMediaPipe();
      if (!mediaPipeReady) return;
      const cameraReady = await startCamera();
      if (!cameraReady) return;
      context.registerUpdatable(processMediaPipeFrame);
    }

    const api = {
      updateConfig: (newConfig) => {
        Object.assign(config, newConfig);
        if (video && 'videoOpacity' in newConfig) {
          video.style.opacity = String(config.videoOpacity);
        }
        if (previewVideo && ('previewOpacity' in newConfig || 'drawHandPreview' in newConfig)) {
          updatePreviewVisibility();
        }
        if ('processWidth' in newConfig || 'processHeight' in newConfig || 'optimized' in newConfig) {
          configureProcessingCanvases();
        }
      },
      getConfig: () => ({ ...config }),
      perfTrace: (traceOptions) => context.mediapipePerfTrace?.(traceOptions),
      isReady: () => isInitialized && context.mediaPipeInitialized
    };

    context.mediaPipeApi = api;
    context.mediaPipeHandsApi = api;
    initialize().catch(console.error);

    return () => {
      context.unregisterUpdatable?.(processMediaPipeFrame);

      if (video?.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
        video.srcObject = null;
      }

      if (previewVideo) {
        previewVideo.srcObject = null;
      }

      if (video?.parentNode) video.parentNode.removeChild(video);
      if (previewVideo?.parentNode) previewVideo.parentNode.removeChild(previewVideo);
      if (poseCanvas?.parentNode) poseCanvas.parentNode.removeChild(poseCanvas);
      if (handCanvas?.parentNode) handCanvas.parentNode.removeChild(handCanvas);
      processCanvas = null;
      processCtx = null;
      deltaCanvas = null;
      deltaCtx = null;
      previousDeltaPixels = null;

      context.setPoseLandmarks?.(null);
      context.setHandLandmarks?.(null);
      context.clearHandFrame?.();
      if (context.setHandFound) context.setHandFound(false);
      context.mediaPipeInitialized = false;
      context.isTracking = false;
      context.mediaPipeApi = null;
      context.mediaPipeHandsApi = null;
    };
  };
}
