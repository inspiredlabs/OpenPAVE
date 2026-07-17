<script>
  // ./routes/hand/MediaPipeHands.svelte
  import { mediapipeVisionHands } from './mediapipeVisionHands.js';
  import { useContext } from './simpleContext.svelte.js';
  
  // Props passed from parent
  let {
    drawSkeletons = true,
    viewportScale = 0.2,
    videoOpacity = 1.0,
    filterAlpha = 0.6,
    numHands = 1,
    // Hand-specific props
    enableHandGestureControl = false,
    maxHandMovementSensitivity = 0.5,
    handOrientationFilterAlpha = 0.7
  } = $props();
  
  const context = useContext();
  
  // Reactive config updates
  $effect(() => {
    const newConfigOptions = {
      drawSkeletons: drawSkeletons,
      viewportScale: viewportScale,
      videoOpacity: videoOpacity,
      filterAlpha: filterAlpha,
      numHands: numHands,
      // Hand-specific options
      enableHandGestureControl: enableHandGestureControl,
      maxHandMovementSensitivity: maxHandMovementSensitivity,
      handOrientationFilterAlpha: handOrientationFilterAlpha
    };
  
    // Update runtime config if API is available
    if (context && context.mediaPipeHandsApi && context.mediaPipeHandsApi.updateConfig) {
      context.mediaPipeHandsApi.updateConfig(newConfigOptions);
    }
  });
  
  // Initial options for factory
  const initialOptionsForFactory = {
    drawSkeletons: drawSkeletons,
    viewportScale: viewportScale,
    videoOpacity: videoOpacity,
    filterAlpha: filterAlpha,
    numHands: numHands,
    enableHandGestureControl: enableHandGestureControl,
    maxHandMovementSensitivity: maxHandMovementSensitivity,
    handOrientationFilterAlpha: handOrientationFilterAlpha
  };
  
  const mediapipeHandsAttachmentFunction = mediapipeVisionHands(initialOptionsForFactory);
  
  </script>
  
  <!-- 
  Attachment point for the MediaPipe Hands logic. 
  `mediapipeVisionHands.js` creates and manages canvas
  for the webcam feed and hand landmark drawing.
  -->
  <aside>
    <div class="selfie hands-tracker" {@attach mediapipeHandsAttachmentFunction}></div>
  </aside>
  
  <style>
  /* 
  NOTE: Video and canvas elements are created and positioned 
  in mediapipeVisionHands.js, not here. This component just 
  provides the attachment point and reactive configuration.
  */
  
  .selfie { 
    transform: scaleX(-1); 
  }
  
  .hands-tracker {
    /* Optional: Add any specific styling for hands tracking container */
    position: relative;
  }
  
  /* Global styles for hand tracking elements (applied by JS in mediapipeVisionHands.js) */
  :global(.hand-canvas) {
    /* These styles are applied in mediapipeVisionHands.js but can be overridden here */
    border-color: #4A90E2 !important; /* Blue border to distinguish from pose */
  }
  
  :global(.selfie.hands-tracker video) {
    /* Video element styling override for hands */
    border-color: #4A90E2 !important;
  }
  
  /* Debug mode indicators */
  :global(.hands-tracker.debug) {
    border: 2px dashed #4A90E2;
  }
  
  /* Optional: Different positioning for hands vs pose */
  .hands-mode {
    /* Could offset position if running both simultaneously */
    /* left: calc(50vw - 7.5rem) !important; */
  }
  </style>