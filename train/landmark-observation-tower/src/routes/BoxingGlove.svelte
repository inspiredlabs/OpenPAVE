<script>
// ./BoxingGlove.svelte

/*
============================================================
The Touch Points for a Self-Updating 3D Component
============================================================
1.  The component is now self-sufficient it doesn't reqire `scene`.
2.  Add `useContext`: To access the shared 3D world (scene, camera) and data (landmarks).
3.  Identify the Data Source: The component is driven by `context.getPoseLandmarks()`.
4.  Add a Differentiator Prop: A `hand` prop ('left' or 'right') tells the component which landmark (15 or 16) to track.
5.  Create a `perFrameUpdate` Function: This contains all the logic that runs every frame—getting data, calculating position, and updating the Three.js object.
6.  Use `$effect` for Setup and Teardown:
    - Setup: Create meshes and call `context.registerUpdatable(perFrameUpdate)`.
    - Teardown: Call `context.unregisterUpdatable(perFrameUpdate)` and dispose of assets.
*/

// This component follows the "Child as Self-Sufficient Agent" pattern.
// 1. It gets the `scene`, `camera`, and raw `poseLandmarks` from the context.
// 2. It does NOT receive position as a prop. It calculates its own position every frame.
// 3. It registers its own `perFrameUpdate` function with the central animation loop.

import * as THREE from 'three';
import { useContext } from './simpleContext.svelte.js';

const context = useContext();

// --- PROPS ---
// Props are now for configuration, not for passing frame-by-frame data.
let {
    hand = 'left', // The key differentiator: 'left' or 'right'
    visible = true,
    scale = 0.05,
    color = 0xff0000,
    viewportScale = 4.0, // Should match LandmarkRenderer for consistent scaling
    distanceInFrontOfCamera = 4.8 // Should match LandmarkRenderer
} = $props();

// --- STATE & UTILITIES ---
let gloveGroup;
let isInitialized = false;
let prevPosition = { x: 0, y: 0, z: 0, initialized: false };

// Landmark indices for the wrist (primary) and shoulder (fallback)
const landmarkIndex = hand === 'left' ? 15 : 16;
const fallbackIndex = hand === 'left' ? 11 : 12;

// Reusable THREE.js objects for performance
const cameraDirection = new THREE.Vector3();
const desiredGroupWorldPosition = new THREE.Vector3();

// --- HELPER FUNCTIONS (Internal to this component) ---
function mapLandmarkToLocalOffset(landmark) {
    if (!landmark) return { x: 0, y: 0, z: 0 };
    const x = -(landmark.x - 0.5) * viewportScale; 
    const y = -(landmark.y - 0.5) * viewportScale;
    const z = landmark.z * viewportScale; 
    return { x, y, z };
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
// This function will be registered to run on every single animation frame.
function perFrameUpdate() {
    if (!isInitialized || !gloveGroup) return;

    // Get live data from the context
    const camera = context.getCamera();
    const landmarks = context.getPoseLandmarks();

    if (!visible || !landmarks) {
        if (gloveGroup.visible) gloveGroup.visible = false;
        return;
    }

    // Fallback logic: Use shoulder if wrist isn't visible
    let targetLandmark = landmarks[landmarkIndex];
    if (!targetLandmark || (targetLandmark.visibility && targetLandmark.visibility < 0.2)) {
        targetLandmark = landmarks[fallbackIndex];
    }
    
    if (!targetLandmark) {
        if (gloveGroup.visible) gloveGroup.visible = false;
        return;
    }
    
    if (!gloveGroup.visible) gloveGroup.visible = true;

    // --- POSITION CALCULATION (Same logic as the working parent controller) ---
    // 1. Calculate the base position in front of the camera
    const camWorldPos = new THREE.Vector3();
    camera.getWorldPosition(camWorldPos);
    camera.getWorldDirection(cameraDirection);
    desiredGroupWorldPosition
        .copy(camWorldPos)
        .add(cameraDirection.multiplyScalar(distanceInFrontOfCamera));

    // 2. Calculate the local offset from the landmark data and add it
    const localOffset = mapLandmarkToLocalOffset(targetLandmark);
    const finalTargetPosition = desiredGroupWorldPosition.add(new THREE.Vector3(localOffset.x, localOffset.y, localOffset.z));

    // 3. Apply smoothing
    const smoothedPosition = applyLowPassFilter(finalTargetPosition, prevPosition);
    prevPosition = { ...smoothedPosition, initialized: true };
    
    // 4. Update the actual Three.js object's position
    gloveGroup.position.copy(smoothedPosition);
}

// --- SETUP & TEARDOWN EFFECT ---
$effect(() => {
    // This effect is now reactive to the scene's existence.
    const scene = context.getScene();
    
    // If the scene isn't ready, this effect stops. It will automatically re-run
    // when main.js provides the scene to the context, triggering a change.
    if (!scene) {
        return; 
    }
    
    // --- SETUP (only runs once the scene is available) ---
    gloveGroup = new THREE.Group();
    gloveGroup.name = `${hand}GloveGroup`;

    const gloveGeometry = new THREE.SphereGeometry(0.3, 18, 12);
    const cuffGeometry = new THREE.CylinderGeometry(0.2, 0.25, 0.3, 12, 1, false);
    const material = new THREE.MeshPhysicalMaterial({ color, roughness: 0.25, metalness: 0.1 });
    
    const gloveMesh = new THREE.Mesh(gloveGeometry, material.clone());
    const cuffMesh = new THREE.Mesh(cuffGeometry, material.clone());
    cuffMesh.material.color.multiplyScalar(0.7);
    cuffMesh.position.y = -0.2;

    gloveGroup.add(gloveMesh);
    //gloveGroup.add(cuffMesh);
    gloveGroup.scale.set(scale, scale, scale);
    
    scene.add(gloveGroup);
    
    // Register the update function to the central animation loop
    context.registerUpdatable(perFrameUpdate);
    isInitialized = true;

    // --- CLEANUP ---
    return () => {
        isInitialized = false;
        context.unregisterUpdatable(perFrameUpdate);
        if (gloveGroup) {
            scene.remove(gloveGroup);
            gloveGroup.traverse((obj) => {
                if (obj.isMesh) {
                    obj.geometry.dispose();
                    obj.material.dispose();
                }
            });
            gloveGroup = null;
        }
    };
});

// --- EFFECT FOR REACTIVE PROP UPDATES ---
// This handles changes to props like `color` or `scale` after initial setup.
$effect(() => {
    if (!gloveGroup) return;
    gloveGroup.scale.set(scale, scale, scale);
    
    const gloveMaterial = gloveGroup.children[0]?.material;
    const cuffMaterial = gloveGroup.children[1]?.material;

    if (gloveMaterial && gloveMaterial.color.getHex() !== color) {
        gloveMaterial.color.set(color);
        if (cuffMaterial) {
            cuffMaterial.color.set(new THREE.Color(color).multiplyScalar(0.7));
        }
    }
});
</script>