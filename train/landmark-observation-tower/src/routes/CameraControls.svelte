<script>
// ./CameraControls.svelte
import * as THREE from 'three';
import { useContext } from './simpleContext.svelte.js';

const context = useContext();

// Position state
let camPosX = $state(0);
let camPosY = $state(0);
let camPosZ = $state(10);

// Sync Yaw & Pitch
let camYaw = $state(context.getCameraYaw ? context.getCameraYaw() : 0);   // For rotation.y
let camPitch = $state(context.getCameraPitch ? context.getCameraPitch() : 0); // For rotation.x
// camRoll (rotation.z) is often kept at 0 for FPS-style controls, but you can add if needed.

let cameraInstance = null;
let checkIntervalId = null;
let uiUpdateRegistered = false;

// Flag to prevent feedback loops when UI updates camera, then camera updates UI
let isUpdatingCameraFromUI = false;

function initializeControls(cam) {
	cameraInstance = cam;
	isUpdatingCameraFromUI = true; // Prevent sync while initializing

	camPosX = cameraInstance.position.x;
	camPosY = cameraInstance.position.y;
	camPosZ = cameraInstance.position.z;
	
	// Init Yaw & Pitch
	camYaw = context.getCameraYaw ? context.getCameraYaw() : cameraInstance.rotation.y;
	camPitch = context.getCameraPitch ? context.getCameraPitch() : cameraInstance.rotation.x;

	isUpdatingCameraFromUI = false;

	if (context && typeof context.registerUpdatable === 'function' && !uiUpdateRegistered) {
		context.registerUpdatable(syncUIFromContextOrCamera);
		uiUpdateRegistered = true;
	}
}

function checkForCameraAndInitialize() { /* ... same as before, using context.camera ... */ 
	if (checkIntervalId) {
			clearInterval(checkIntervalId);
			checkIntervalId = null;
	}
	const camFromContext = context.camera; // Using direct access as per your context
	if (camFromContext) {
			initializeControls(camFromContext);
	} else if (context) {
			checkIntervalId = setInterval(() => {
					const polledCam = context.camera;
					if (polledCam) {
							clearInterval(checkIntervalId);
							checkIntervalId = null;
							initializeControls(polledCam);
					}
			}, 200);
	}
}

// Syncs UI FROM the context's shared yaw/pitch, or directly from camera
function syncUIFromContextOrCamera() {
	if (isUpdatingCameraFromUI) return; // Don't sync if UI just caused the change

	if (cameraInstance) { // Position sync remains direct from camera
			const posThreshold = 0.005;
			if (Math.abs(camPosX - cameraInstance.position.x) > posThreshold) camPosX = cameraInstance.position.x;
			if (Math.abs(camPosY - cameraInstance.position.y) > posThreshold) camPosY = cameraInstance.position.y;
			if (Math.abs(camPosZ - cameraInstance.position.z) > posThreshold) camPosZ = cameraInstance.position.z;
	}

	// Rotation syncs from context's shared values
	const rotThreshold = 0.005; // Radians
	const contextYaw = context.getCameraYaw ? context.getCameraYaw() : 0;
	const contextPitch = context.getCameraPitch ? context.getCameraPitch() : 0;

	if (Math.abs(camYaw - contextYaw) > rotThreshold) {
			camYaw = contextYaw;
	}
	if (Math.abs(camPitch - contextPitch) > rotThreshold) {
			camPitch = contextPitch;
	}
}

$effect(() => { // Mount/Unmount
	checkForCameraAndInitialize();
	return () => {
		if (checkIntervalId) clearInterval(checkIntervalId);
		if (uiUpdateRegistered && context && context.unregisterUpdatable) {
			context.unregisterUpdatable(syncUIFromContextOrCamera);
			uiUpdateRegistered = false;
		}
		cameraInstance = null;
	};
});

// Effect to update camera POSITION from UI
$effect(() => {
	if (cameraInstance) {
		isUpdatingCameraFromUI = true;
		cameraInstance.position.set(camPosX, camPosY, camPosZ);
		isUpdatingCameraFromUI = false;
	}
});

// Effect to update CONTEXT's YAW and PITCH from UI sliders
// doomControls will then pick these up to rotate the camera.
$effect(() => {
	if (context.setCameraYaw) {
		isUpdatingCameraFromUI = true; // Signal that UI is making a change
		context.setCameraYaw(camYaw);
		isUpdatingCameraFromUI = false;
	}
});

$effect(() => {
	if (context.setCameraPitch) {
		isUpdatingCameraFromUI = true; // Signal that UI is making a change
		context.setCameraPitch(camPitch); // Context setter will clamp
		isUpdatingCameraFromUI = false;
	}
});

</script>

<small class="camera-controls-panel">
  <h2>FpsCamera</h2>
  
  <div class="control-section">
    <strong>Position:</strong>
    <div class="control-group">
      <label>X:<span>{camPosX.toFixed(1)}</span>
	      <input type="range" min="-20" max="20" step="0.1" bind:value={camPosX} />
			</label>
    </div>
    <div class="control-group">
      <label>Y:<span>{camPosY.toFixed(1)}</span>
	      <input type="range" min="-20" max="20" step="0.1" bind:value={camPosY} />
			</label>
    </div>
    <div class="control-group">
      <label>Z:<span>{camPosZ.toFixed(1)}</span>
	      <input type="range" min="-20" max="20" step="0.1" bind:value={camPosZ} />
			</label>
    </div>
  </div>

  <div class="control-section">
    <strong>Orientation (FPS Style):</strong>
    <div class="control-group">
      <label>Yaw (Y Rot):<span>{camYaw.toFixed(2)}</span>
	      <input type="range" min={-Math.PI} max={Math.PI} step="0.01" bind:value={camYaw} />
			</label>
    </div>
    <div class="control-group">
      <label>Pitch (X Rot):<span>{camPitch.toFixed(2)}</span>
	      <input type="range" min={-Math.PI/2 + 0.01} max={Math.PI/2 - 0.01} step="0.01" bind:value={camPitch} />
			</label>
    </div>
    
  </div>
</small>

<style>
small {
  bottom: 220px;
}
</style>