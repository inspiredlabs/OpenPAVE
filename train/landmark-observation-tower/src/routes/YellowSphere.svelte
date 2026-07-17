<script>
// ./YellowSphere.svelte
import * as THREE from 'three';
import { useContext } from './simpleContext.svelte.js';

let {
	radius = 0.25, // Radius of the visual yellow sphere AND its collider
	color = 0xFFFF00,
	// initialPosition is less relevant now as it's tied to camera
	distanceInFrontOfCamera = 0, // How far the center of the sphere is from camera
	showBoundingSphere = false // NEW: Prop to toggle wireframe visibility
} = $props();

const context = useContext();
let yellowSphereMesh = null;
let cameraInstance = null;

// --- NEW: Vector3 to reuse for getting world positions ---
const targetSphereWorldPosition = new THREE.Vector3();
	
// --- NEW: Wireframe for the bounding sphere ---
let boundingSphereWireframe = null;

// For placing in front of camera
const cameraDirection = new THREE.Vector3();
const desiredSphereWorldPosition = new THREE.Vector3(); // Renamed for clarity

// The logical bounding sphere for collision detection
// Its center will be updated to match the mesh's world position
const yellowSphereCollider = new THREE.Sphere(new THREE.Vector3(), radius);
let gameStarted = $state(context.gameStarted ?? false); // Assuming gameStarted is in context
let initialYellowSphereY = $state(null); // To store the initial Y position after camera setup
	
function updateYellowSphereAndColliderPosition() {
	if (!cameraInstance) { // Ensure camera is available
			cameraInstance = context.getCamera();
			if (!cameraInstance) return; // Still not available, skip update
	}
	if (!yellowSphereMesh) return;


	// Get camera's world position and direction
	const camWorldPos = new THREE.Vector3(); // Not strictly needed if we only use direction
	cameraInstance.getWorldPosition(camWorldPos); // Not strictly needed if we only use direction
	cameraInstance.getWorldDirection(cameraDirection); // Fills cameraDirection with the direction

	// Calculate the desired world position for the sphere
	desiredSphereWorldPosition
		.copy(camWorldPos)
		.add(cameraDirection.multiplyScalar(distanceInFrontOfCamera));
	desiredSphereWorldPosition.x = 0;
	
	// Update the visual mesh position
	yellowSphereMesh.position.copy(desiredSphereWorldPosition);

	// --- CRITICAL: Update the collider's center to the new world position of the mesh ---
	// If the mesh has any parents that affect its world position,
	// yellowSphereMesh.getWorldPosition(yellowSphereCollider.center) is safest.
	// Since we add it directly to the scene, .position is its world position.
	yellowSphereCollider.center.copy(yellowSphereMesh.position);

	// Update wireframe position if it exists
	if (boundingSphereWireframe) {
		boundingSphereWireframe.position.copy(yellowSphereMesh.position);
	}
}

$effect(() => { // Main setup for visual mesh and wireframe
	const sceneInstance = context.getScene();
	cameraInstance = context.getCamera();

	if (!sceneInstance) {
		console.warn("[YellowSphere] Scene not available from context for setup.");
		return; // Don't proceed if scene isn't there
	}

	// Visual Yellow Sphere
	const visualGeometry = new THREE.SphereGeometry(radius, 16, 12);
	const visualMaterial = new THREE.MeshStandardMaterial({
		color: color,
		emissive: color,
		emissiveIntensity: 0.5,
		metalness: 0.2,
		roughness: 0.5
	});
	yellowSphereMesh = new THREE.Mesh(visualGeometry, visualMaterial);
	// Initial position will be quickly updated by updateYellowSphereAndColliderPosition
	yellowSphereMesh.position.set(0,0,0); // Initial off-screen guess -distanceInFrontOfCamera
	sceneInstance.add(yellowSphereMesh);

	// --- NEW: Setup Bounding Sphere Wireframe ---
	if (showBoundingSphere) {
		const wireframeGeometry = new THREE.SphereGeometry(radius, 16, 12); // Same radius
		const wireframeMaterial = new THREE.MeshBasicMaterial({
			color: 0x00ff00, // Green wireframe for contrast
			wireframe: true,
			transparent: true,
			opacity: 0.5
		});
		boundingSphereWireframe = new THREE.Mesh(wireframeGeometry, wireframeMaterial);
		// It will be positioned with yellowSphereMesh in the update loop
		sceneInstance.add(boundingSphereWireframe);
	}
	
	// Update collider radius if the prop changes (though radius isn't $state here, good practice if it were)
	yellowSphereCollider.radius = radius;


	// Register perFrameUpdate (which includes collision checks)
	let isRegistered = false;
	function checkAndRegister() {
			if (!cameraInstance) cameraInstance = context.getCamera();
			if (cameraInstance && yellowSphereMesh) {
					context.registerUpdatable(perFrameUpdate);
					isRegistered = true;
					// console.log("[YellowSphere] Registered perFrameUpdate.");
					return true;
			}
			return false;
	}

	if (!checkAndRegister()) {
			const checkInterval = setInterval(() => {
					if (checkAndRegister()) {
							clearInterval(checkInterval);
					}
			}, 200);
	}

	return () => { // Cleanup
		if (yellowSphereMesh) {
			sceneInstance?.remove(yellowSphereMesh);
			yellowSphereMesh.geometry.dispose();
			yellowSphereMesh.material.dispose();
			yellowSphereMesh = null;
		}
		if (boundingSphereWireframe) { // NEW: Cleanup wireframe
			sceneInstance?.remove(boundingSphereWireframe);
			boundingSphereWireframe.geometry.dispose();
			boundingSphereWireframe.material.dispose();
			boundingSphereWireframe = null;
		}
		if (isRegistered) { // Only unregister if it was registered
				context.unregisterUpdatable(perFrameUpdate);
		}
	};
});

// Reactive effect for toggling wireframe visibility if showBoundingSphere prop changes
$effect(() => {
	if (boundingSphereWireframe) {
		boundingSphereWireframe.visible = showBoundingSphere;
	}
	// If it's turned on after initial setup, and wasn't created, we might need to create it here too.
	// For simplicity, the above $effect handles initial creation. This just handles visibility toggle.
	// A more robust solution would create/destroy it if the prop changes from false to true *after* initial mount.
	// However, usually, this prop is set once.
});


let collisionCheckCounter = 0;
const COLLISION_CHECK_INTERVAL = 2; // Check more frequently if needed

function perFrameUpdate(deltaTime) {
	updateYellowSphereAndColliderPosition(); // This now updates visual mesh, collider center, and wireframe pos

	collisionCheckCounter++;
	if (collisionCheckCounter % COLLISION_CHECK_INTERVAL === 0) {
			performCollisionChecks();
	}
}

function performCollisionChecks() {
    if (!yellowSphereMesh || !context || !yellowSphereCollider.radius) return;

    const collidableSpheres = context.getCollidableSpheres ? context.getCollidableSpheres() : [];
    
    for (const columnSphereData of collidableSpheres) {
        if (!columnSphereData || !columnSphereData.mesh || !columnSphereData.mesh.visible || columnSphereData.isHit) continue;

        // --- CRITICAL CHANGE: Get the WORLD position of the target sphere's mesh ---
        columnSphereData.mesh.getWorldPosition(targetSphereWorldPosition); 
        /* Local Position algo:
				 * This method calculates the mesh's absolute position in world space,
				 * taking into account its local position and the transformations
				 * (position, rotation, scale) of all its parent objects in the scene graph
				 * (including the gridGroupInstance from SphereColumn which is placed at
				 * basePosition).
				 */

        // Use this world position for creating the temporary collider for the target
        const targetSphereCollider = new THREE.Sphere(targetSphereWorldPosition, columnSphereData.radius);
        
        // Optional: Logging for debug
				// NOT IMPLEMENTED: You'll need: componentInstanceId (which becomes parentId) of your magenta and green SphereColumn instances.
        // if (columnSphereData.parentId === /* id of your green column if known */ || columnSphereData.parentId === /* id of magenta */) {
        //     const dist = yellowSphereCollider.center.distanceTo(targetSphereCollider.center);
        //     const combinedR = yellowSphereCollider.radius + targetSphereCollider.radius;
        //     console.log(
        //         `Checking Yellow (Z:${yellowSphereCollider.center.z.toFixed(1)}) against Target (Parent:${columnSphereData.parentId.substring(0,4)}, ID:${columnSphereData.id.substring(0,4)}, WorldZ:${targetSphereCollider.center.z.toFixed(1)}), Dist:${dist.toFixed(1)}, RadiiSum:${combinedR.toFixed(1)}, Intersects: ${yellowSphereCollider.intersectsSphere(targetSphereCollider)}`
        //     );
        // }


        if (yellowSphereCollider.intersectsSphere(targetSphereCollider)) {
            // console.log(`    COLLISION! TargetID: ${columnSphereData.id.substring(0,4)}, Parent: ${columnSphereData.parentId.substring(0,4)}`);
            if (typeof columnSphereData.onHit === 'function') {
                columnSphereData.onHit();
            }
        }
    }
}

</script>