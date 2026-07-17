<script>
// ./FloatingBoxes.svelte
/**
* This component should follow a declarative pattern.
* It manages its own objects and collision logic internally,
* driven by props from the parent. It should be self-contained.
* 
* 1. Dependencies as Props: The component declares everything it 
* needs from the outside world (scene or data stream) as a prop.
* 2. Lifecycle Management $effect hook: primary $effect used to
* create and add objects to scene when the component mounts.
* This same effect returns a cleanup function to dispose of resources.
* 3. Reactive Updates: $effect hooks watch for changes in props, like:
* landmarks or left/rightGloveBoundingSphere).
* When props change, it runs automatically updating it's internal state.
0xaaaaaa
*/

// This component follows the "Child as Self-Sufficient Agent" pattern.
// 1. It gets `scene`, `camera`, and `poseLandmarks` from the context.
// 2. It internally creates and manages the bounding spheres for the gloves.
// 3. It registers its own update function to the central animation loop to handle collisions.
// ./FloatingBoxes.svelte
// This component follows the "Child as Self-Sufficient Agent" pattern.

import * as THREE from 'three';
import { useContext } from './simpleContext.svelte.js';

const context = useContext();

// --- PROPS ---
// Props are for configuration, not for passing live data.
let {
   boxGridZ = -0.6, // Start the grid a bit further back
   //gridSize = 3,
   //boxSpacing = 0.15,
   //boxSize = 0.08,
   viewportScale = 4.0, // Must match other agents for consistent positioning
   distanceInFrontOfCamera = 4.8
} = $props();

/* Svelte Reactivity */
let gridSize = $state(3); // how many boxes?
let yOffset = $state(3); // how far off the ground?
let boxSpacing = $state(2.5); // gaps?
let boxSize = $state(2); // It works!

	
// --- INTERNAL STATE ---
let targetBoxes = [];
let boxCollisionCooldowns = [];
const COLLISION_COOLDOWN = 150; // in frames
let isInitialized = false;

// The component now owns the colliders for the gloves.
let leftGloveCollider = new THREE.Sphere(new THREE.Vector3(), 0.1); // Radius can be tuned
let rightGloveCollider = new THREE.Sphere(new THREE.Vector3(), 0.1);

// State for smoothing glove positions
let prevLeftGlovePos = { x: 0, y: 0, z: 0, initialized: false };
let prevRightGlovePos = { x: 0, y: 0, z: 0, initialized: false };

// --- HELPER FUNCTIONS ---
function mapLandmarkToLocalOffset(landmark) {
   if (!landmark) return null;
   return {
       x: -(landmark.x - 0.5) * viewportScale, // mirror to match pose-lite input
       y: -(landmark.y - 0.5) * viewportScale, // invert to match ThreeJS coords
       z: landmark.z * viewportScale,
   };
}

function applyLowPassFilter(newValue, prevValue, alpha = 0.7) {
   if (!prevValue.initialized || !newValue) return newValue || prevValue;
   const mappedAlpha = alpha * alpha;
   return {
       x: mappedAlpha * newValue.x + (1 - mappedAlpha) * prevValue.x,
       y: mappedAlpha * newValue.y + (1 - mappedAlpha) * prevValue.y,
       z: mappedAlpha * newValue.z + (1 - mappedAlpha) * prevValue.z
   };
}

// --- PER-FRAME UPDATE LOGIC ---
function perFrameUpdate() {
   if (!isInitialized) return;

   const camera = context.getCamera();
   const landmarks = context.getPoseLandmarks();
   if (!camera || !landmarks) return;

   // --- 1. Update Glove Collider Positions ---
   const leftWrist = landmarks[15];
   const rightWrist = landmarks[16];

   // Get the base position in front of the camera
   const camWorldPos = new THREE.Vector3();
   camera.getWorldPosition(camWorldPos);
   const cameraDirection = new THREE.Vector3();
   camera.getWorldDirection(cameraDirection);
   const basePosition = camWorldPos.add(cameraDirection.multiplyScalar(distanceInFrontOfCamera));
   
   // Calculate and smooth the world position for each glove collider
   let leftOffset = mapLandmarkToLocalOffset(leftWrist);
   let rightOffset = mapLandmarkToLocalOffset(rightWrist);

   if (leftOffset) {
       let finalLeftPos = new THREE.Vector3(basePosition.x + leftOffset.x, basePosition.y + leftOffset.y, basePosition.z + leftOffset.z);
       let smoothedLeft = applyLowPassFilter(finalLeftPos, prevLeftGlovePos);
       leftGloveCollider.center.copy(smoothedLeft);
       prevLeftGlovePos = { ...smoothedLeft, initialized: true };
   }
   if (rightOffset) {
       let finalRightPos = new THREE.Vector3(basePosition.x + rightOffset.x, basePosition.y + rightOffset.y, basePosition.z + rightOffset.z);
       let smoothedRight = applyLowPassFilter(finalRightPos, prevRightGlovePos);
       rightGloveCollider.center.copy(smoothedRight);
       prevRightGlovePos = { ...smoothedRight, initialized: true };
   }

   // --- 2. Check for Collisions ---
   for (let i = 0; i < targetBoxes.length; i++) {
       if (boxCollisionCooldowns[i] > 0) {
           boxCollisionCooldowns[i]--;
           continue;
       }

       const box = targetBoxes[i];
       const boxCollider = new THREE.Box3().setFromObject(box);

       if (boxCollider.intersectsSphere(leftGloveCollider)) {
           box.material.color.set(0xff0000); // Red
           boxCollisionCooldowns[i] = COLLISION_COOLDOWN;
       } else if (boxCollider.intersectsSphere(rightGloveCollider)) {
           box.material.color.set(0x0000ff); // Blue
           boxCollisionCooldowns[i] = COLLISION_COOLDOWN;
       } else if (box.material.color.getHex() !== 0xaaaaaa) {
           box.material.color.set(0xaaaaaa);
       }
   }
}


// --- SETUP & TEARDOWN EFFECT ---
$effect(() => {
   const scene = context.getScene();
   if (!scene) {
       return; // Wait for the scene to be ready
   }

   const boxGeometry = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
   const defaultBoxMaterial = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, roughness: 0.6 });

   for (let row = 0; row < gridSize; row++) {
       for (let col = 0; col < gridSize; col++) {
           const x = (col - Math.floor(gridSize / 2)) * boxSpacing;
           const y = (row - Math.floor(gridSize / 2)) * boxSpacing + 1.2; // Offset Y to be in view
           const box = new THREE.Mesh(boxGeometry, defaultBoxMaterial.clone());
           box.position.set(x, y, boxGridZ);
           scene.add(box);
           targetBoxes.push(box);
       }
   }
   boxCollisionCooldowns = new Array(targetBoxes.length).fill(0);
   
   // Register the update function
   context.registerUpdatable(perFrameUpdate);
   isInitialized = true;

   // --- Cleanup Logic ---
   return () => {
       isInitialized = false;
       context.unregisterUpdatable(perFrameUpdate);
       targetBoxes.forEach(box => {
           if (box?.parent) {
               scene.remove(box);
               box.material?.dispose();
           }
       });
       boxGeometry?.dispose();
       defaultBoxMaterial?.dispose();
       targetBoxes = [];
   };
});

// --- EFFECT FOR REACTIVE PROP UPDATES ---
$effect(() => {
   // This effect runs only when boxGridZ prop changes.
   if (!isInitialized) return;
   targetBoxes.forEach(box => {
       if (box) box.position.z = boxGridZ;
   });
});

</script>

<!-- UI PANEL - This HTML is now part of the component -->
<small class="panel">
   <h3>Floating Boxes</h3>
   <div class="control-group">
			<label>Y Offset: <span>{yOffset.toFixed(2)}</span>
				<input type="range" min="0" max="5" step="0.1" bind:value={yOffset} />
			</label>
			<label>boxSpacing: <span>{boxSpacing.toFixed(2)}</span>
				<input type="range" min="0" max="5" step="0.1" bind:value={boxSpacing} />
			</label>
			<label>boxSize: <span>{boxSize.toFixed(2)}</span>
				<input type="range" min="1" max="5" step="0.1" bind:value={boxSize} />
			</label>
   </div>
</small>

<style>
.panel {
 bottom: 20px; /* Position it opposite GizmoControls */
 left: 20px;
}
</style>