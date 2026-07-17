<script>
// ./PickingControls.svelte
/**
 * Precision "Index-Pinch" Interaction
 * 
 * 1. Aiming: Raycast from Camera -> Index Finger Tip (Landmark 8).
 * 2. Trigger: STRICT Thumb (4) + Index (8) distance.
 *    - Prevents accidental grabs from relaxed hands (false positives).
 * 3. Feedback: 
 *    - Yellow: Idle
 *    - White:  Pinch Detected (Air)
 *    - Cyan:   Hovering Object
 *    - Green:  Grabbing
 */

import * as THREE from 'three';
import { useContext } from './simpleContext.svelte.js';

// --- FAT LINE IMPORTS ---
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';

const context = useContext();

// --- UX TUNING ---
const VIEWPORT_SCALE = 6.0;       
const HAND_DEPTH_RATIO = 0.4;     
const TARGET_PLANE_Z = 0.0;       

// --- PRECISION TUNING ---
// World Unit Distances (Scale 8.0)
// 0.35 requires tips to physically touch or overlap slightly.
// 0.50 allows release without opening hand fully.
const PINCH_THRESHOLD_START = 0.30; 
// We set END to 0.15 to keep it "sticky" but drop immediately 
// when you visibly separate them.
const PINCH_THRESHOLD_END = 0.34; 

const FILTER_ALPHA = 1.0;           

// --- STATE ---
let isGrabbing = $state(false);
let isPinching = $state(false);
let grabInputType = $state(null); 

// Physics
let grabDistance = $state(0);
let canvasElement = $state(null);

// Hand
const MAX_HAND_LANDMARKS = 21;
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [17, 0]
];
const PALM_TRIANGLES = [
    [0, 1, 5],
    [1, 2, 5],
    [0, 5, 9],
    [0, 9, 13],
    [0, 13, 17]
];
let handGroup = null;
let landmarkMeshes = [];
let prevPositions = [];
let boneMeshes = [];
let palmMesh = null;
let palmPositionAttribute = null;

// Visuals
let targetLine = null; 
let currentHighlight = null;       
let currentHighlightTarget = null; 

// Math
const _cameraPos = new THREE.Vector3();
const _cameraDir = new THREE.Vector3();
const _targetPos = new THREE.Vector3();
const _aimPoint = new THREE.Vector3(); // Index Tip
const _thumbPos = new THREE.Vector3(); // Thumb Tip
const _rayDir = new THREE.Vector3();
const _boneDirection = new THREE.Vector3();
const _boneMidpoint = new THREE.Vector3();
const _cylinderAxis = new THREE.Vector3(0, 1, 0);

let raycaster = null; 


// --- 1. SETUP VISUALS ---
$effect(() => {
    const scene = context.getScene();
    if (!scene) return;

    raycaster = new THREE.Raycaster();

    handGroup = new THREE.Group();
    handGroup.name = 'PickingHandGroup';
    handGroup.renderOrder = 999;
    scene.add(handGroup);

    // Landmarks
    const geometry = new THREE.SphereGeometry(0.04, 8, 8);
    const materialDefault = new THREE.MeshBasicMaterial({ 
        color: 0x4A90E2, 
        depthTest: false, 
        transparent: true, 
        opacity: 0.6 
    });
    const materialTip = new THREE.MeshBasicMaterial({ 
        color: 0xFFFF00, 
        depthTest: false, 
        transparent: true, 
        opacity: 0.6 
    });

    for (let i = 0; i < MAX_HAND_LANDMARKS; i++) {
        // Only visualize Thumb(4) and Index(8) as "Active" tips
        const isTip = (i === 4 || i === 8);
        const mesh = new THREE.Mesh(geometry, isTip ? materialTip.clone() : materialDefault);
        
        if (isTip) mesh.scale.set(1.5, 1.5, 1.5);
        mesh.visible = false;
        mesh.userData.ignoreRaycast = true; 
        
        handGroup.add(mesh);
        landmarkMeshes.push(mesh);
        prevPositions.push({ x: 0, y: 0, z: 0, initialized: false });
    }

    // Skeleton visuals share this exact handGroup and read the already-mapped
    // landmark mesh positions. There is no second projection or smoothing pass.
    const boneGeometry = new THREE.CylinderGeometry(0.025, 0.025, 1, 8);
    const boneMaterial = new THREE.MeshBasicMaterial({
        color: 0x4A90E2,
        depthTest: false,
        transparent: true,
        opacity: 0.72
    });
    for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
        const bone = new THREE.Mesh(boneGeometry, boneMaterial);
        bone.name = `handBone-${startIndex}-${endIndex}`;
        bone.renderOrder = 998;
        bone.userData.ignoreRaycast = true;
        handGroup.add(bone);
        boneMeshes.push(bone);
    }

    const palmGeometry = new THREE.BufferGeometry();
    palmPositionAttribute = new THREE.BufferAttribute(
        new Float32Array(PALM_TRIANGLES.length * 3 * 3),
        3
    );
    palmPositionAttribute.setUsage(THREE.DynamicDrawUsage);
    palmGeometry.setAttribute('position', palmPositionAttribute);

    const palmMaterial = new THREE.MeshBasicMaterial({
        color: 0x4A90E2,
        depthTest: false,
        depthWrite: false,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide
    });
    palmMesh = new THREE.Mesh(palmGeometry, palmMaterial);
    palmMesh.name = 'handPalm';
    palmMesh.renderOrder = 997;
    palmMesh.frustumCulled = false;
    palmMesh.userData.ignoreRaycast = true;
    handGroup.add(palmMesh);

    // Aim Line
    const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const lineMat = new THREE.LineBasicMaterial({
        color: 0x00FF00, transparent: true, opacity: 0.5
    });
    targetLine = new THREE.Line(lineGeo, lineMat);
    targetLine.frustumCulled = false;
    targetLine.visible = false;
    scene.add(targetLine);

    return () => {
        if (scene) {
            scene.remove(handGroup);
            scene.remove(targetLine);
        }
        clearHighlight();
        geometry.dispose();
        materialDefault.dispose();
        materialTip.dispose();
        boneGeometry.dispose();
        boneMaterial.dispose();
        palmGeometry.dispose();
        palmMaterial.dispose();
        lineGeo.dispose();
        lineMat.dispose();
        landmarkMeshes = [];
        prevPositions = [];
        boneMeshes = [];
        palmMesh = null;
        palmPositionAttribute = null;
        handGroup = null;
        raycaster = null;
    };
});


// --- 2. HIGHLIGHT SYSTEM ---

function updateHighlight(targetMesh) {
    // 1. NEW: Explicitly skip static bodies
    if (targetMesh && targetMesh.userData?.isStatic) {
        clearHighlight();
        return;
    }

    // 2. Existing logic...
    if (currentHighlightTarget === targetMesh) {
        if (currentHighlight && currentHighlight.material) {
                currentHighlight.material.resolution.set(context.canvasWidth, context.canvasHeight);
        }
        return;
    }

    clearHighlight();

    if (targetMesh) {
        currentHighlightTarget = targetMesh;
        const edgesGeo = new THREE.EdgesGeometry(targetMesh.geometry, 15); 
        const lineGeo = new LineGeometry();
        lineGeo.setPositions(edgesGeo.attributes.position.array);
        
        const lineMat = new LineMaterial({
            color: 0x00AAAA, 
            linewidth: 1.5, 
            resolution: new THREE.Vector2(context.canvasWidth, context.canvasHeight),
            dashed: false,
            alphaToCoverage: true, 
            depthTest: false,      
            transparent: true,
            opacity: 0.8
        });

        currentHighlight = new LineSegments2(lineGeo, lineMat);
        currentHighlight.scale.copy(targetMesh.scale).multiplyScalar(1.001); 
        currentHighlight.renderOrder = 1000;
        currentHighlight.userData.ignoreRaycast = true; 
        
        targetMesh.add(currentHighlight);
        edgesGeo.dispose();
    }
}

function clearHighlight() {
    if (currentHighlight && currentHighlightTarget) {
        currentHighlightTarget.remove(currentHighlight);
        currentHighlight.geometry.dispose();
        currentHighlight.material.dispose();
        currentHighlight = null;
        currentHighlightTarget = null;
    }
}


// --- 3. HELPERS ---

function mapLandmark(landmark) {
    return {
        x: -(landmark.x - 0.5) * VIEWPORT_SCALE,
        y: -(landmark.y - 0.5) * VIEWPORT_SCALE,
        z: landmark.z * VIEWPORT_SCALE
    };
}

function applyFilter(newVal, prevVal, alpha) {
    if (!prevVal.initialized) return { ...newVal, initialized: true };
    return {
        x: prevVal.x * (1 - alpha) + newVal.x * alpha,
        y: prevVal.y * (1 - alpha) + newVal.y * alpha,
        z: prevVal.z * (1 - alpha) + newVal.z * alpha,
        initialized: true
    };
}

function setTipsColor(hexColor) {
    // Only color Thumb and Index
    if (landmarkMeshes[4]) landmarkMeshes[4].material.color.setHex(hexColor);
    if (landmarkMeshes[8]) landmarkMeshes[8].material.color.setHex(hexColor);
}

function updateSkeleton() {
    for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
        const [startIndex, endIndex] = HAND_CONNECTIONS[i];
        const start = landmarkMeshes[startIndex];
        const end = landmarkMeshes[endIndex];
        const bone = boneMeshes[i];

        if (!start?.visible || !end?.visible) {
            bone.visible = false;
            continue;
        }

        _boneDirection.subVectors(end.position, start.position);
        const length = _boneDirection.length();
        if (length < 0.0001) {
            bone.visible = false;
            continue;
        }

        bone.visible = true;
        _boneMidpoint.addVectors(start.position, end.position).multiplyScalar(0.5);
        bone.position.copy(_boneMidpoint);
        bone.quaternion.setFromUnitVectors(_cylinderAxis, _boneDirection.normalize());
        bone.scale.set(1, length, 1);
    }

    let vertex = 0;
    for (const triangle of PALM_TRIANGLES) {
        for (const landmarkIndex of triangle) {
            const position = landmarkMeshes[landmarkIndex].position;
            palmPositionAttribute.setXYZ(vertex, position.x, position.y, position.z);
            vertex += 1;
        }
    }
    palmPositionAttribute.needsUpdate = true;
    palmMesh.visible = landmarkMeshes.every((mesh) => mesh.visible);
}


// --- 4. PHYSICS ---

function performAssistRaycast(camera, aimPoint) {
    _rayDir.subVectors(aimPoint, camera.position).normalize();
    raycaster.set(camera.position, _rayDir);
    
    const scene = context.getScene();
    const intersects = raycaster.intersectObjects(scene.children, true);
    
    for (const hit of intersects) {
        const obj = hit.object;
        if (obj.userData?.ignoreRaycast) continue;
        if (obj.parent === handGroup) continue;
        if (!obj.visible) continue;

        if (obj.userData?.isPhysicsBody && 
            !obj.userData?.isStatic && 
            obj.userData?.physicsBodyId !== undefined) {
            return hit;
        }
    }
    return null;
}

function startDrag(bodyId, hitPoint) {
    const physics = context.getAVBDPhysics();
    if (physics?.world.startDrag(bodyId, hitPoint.x, hitPoint.y)) {
        isGrabbing = true;
        if (context.orbitControls) context.orbitControls.enabled = false;
        return true;
    }
    return false;
}

function updateDrag(camera, aimPoint) {
    _rayDir.subVectors(aimPoint, camera.position).normalize();
    const targetPos = camera.position.clone().add(_rayDir.multiplyScalar(grabDistance));
    
    const physics = context.getAVBDPhysics();
    if (physics?.world) {
        physics.world.updateDrag(targetPos.x, targetPos.y);
    }

    if (targetLine) {
        const positions = targetLine.geometry.attributes.position.array;
        aimPoint.toArray(positions, 0);
        targetPos.toArray(positions, 3);
        targetLine.geometry.attributes.position.needsUpdate = true;
        targetLine.visible = true;
    }
    
    if (currentHighlight && currentHighlight.material) {
        currentHighlight.material.resolution.set(context.canvasWidth, context.canvasHeight);
    }
}

function endDrag() {
    const physics = context.getAVBDPhysics();
    physics?.world.endDrag();
    
    isGrabbing = false;
    grabInputType = null;
    
    if (context.orbitControls) context.orbitControls.enabled = true;
    if (targetLine) targetLine.visible = false;
    
    setTipsColor(0xFFFF00);
}


// --- 5. MAIN LOOP ---

function updateHandLoop() {
    const landmarks = context.getHandLandmarks();
    const camera = context.getCamera();
    
    if (!landmarks || !camera || !handGroup || landmarks.length === 0) {
        handGroup.visible = false;
        if (isGrabbing && grabInputType === 'hand') endDrag();
        clearHighlight();
        return;
    }

    handGroup.visible = true;

    // A. FLUSH TRANSFORMS
    camera.updateMatrixWorld();

    // B. POSITION HAND
    camera.getWorldPosition(_cameraPos);
    camera.getWorldDirection(_cameraDir);
    const targetPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -TARGET_PLANE_Z);
    const totalDist = targetPlane.distanceToPoint(_cameraPos);
    const handDist = totalDist * HAND_DEPTH_RATIO;

    _targetPos.copy(_cameraPos).add(_cameraDir.multiplyScalar(handDist));
    handGroup.position.copy(_targetPos);
    handGroup.quaternion.copy(camera.quaternion);
    
    handGroup.updateMatrixWorld(true); // Force update

    // C. UPDATE LANDMARKS
    for (let i = 0; i < MAX_HAND_LANDMARKS; i++) {
        if (!landmarks[i]) {
            landmarkMeshes[i].visible = false;
            continue;
        }
        const raw = mapLandmark(landmarks[i]);
        const smooth = applyFilter(raw, prevPositions[i], FILTER_ALPHA);
        prevPositions[i] = smooth;
        landmarkMeshes[i].position.set(smooth.x, smooth.y, smooth.z);
        landmarkMeshes[i].visible = true;
    }
    updateSkeleton();
    handGroup.updateMatrixWorld(true);

    if (isGrabbing && grabInputType === 'mouse') return;

    // D. PRECISE PINCH LOGIC (Thumb vs Index Only)
    landmarkMeshes[4].getWorldPosition(_thumbPos); // Thumb
    landmarkMeshes[8].getWorldPosition(_aimPoint); // Index (Also used for Aiming)
    
    const pinchDist = _thumbPos.distanceTo(_aimPoint);
    
    // Hysteresis
    if (isPinching) {
        if (pinchDist > PINCH_THRESHOLD_END) isPinching = false;
    } else {
        if (pinchDist < PINCH_THRESHOLD_START) isPinching = true;
    }

    // E. INTERACTION
    if (!isGrabbing) {
        // HOVER
        const hit = performAssistRaycast(camera, _aimPoint);
        
        if (hit) {
            updateHighlight(hit.object);
            
            // Visual State: Are we pinching (White) or Ready (Cyan)?
            if (isPinching) {
                // START GRAB
                grabDistance = hit.distance;
                if (startDrag(hit.object.userData.physicsBodyId, hit.point)) {
                    grabInputType = 'hand';
                    setTipsColor(0x00FF00); // Green
                }
            } else {
                setTipsColor(0x00FFFF); // Cyan (Hover)
            }
        } else {
            clearHighlight();
            // Visual State: Pinching Air (White) or Idle (Yellow)
            setTipsColor(isPinching ? 0xFFFFFF : 0xFFFF00);
        }

    } else if (isGrabbing && isPinching) {
        // DRAG
        updateDrag(camera, _aimPoint);
        setTipsColor(0x00FF00); 
    } else {
        // RELEASE
        endDrag();
    }
}

$effect(() => {
    context.registerUpdatable(updateHandLoop);
    return () => context.unregisterUpdatable(updateHandLoop);
});


// --- MOUSE FALLBACK ---

function getMouseNDC(e) {
    const rect = canvasElement.getBoundingClientRect();
    return {
        x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
        y: -((e.clientY - rect.top) / rect.height) * 2 + 1
    };
}

function handleMouseDown(e) {
    if (e.button !== 0 || isGrabbing) return;
    const ndc = getMouseNDC(e);
    const camera = context.getCamera();
    if (!camera) return;

    raycaster.setFromCamera(ndc, camera);
    const intersects = raycaster.intersectObjects(context.getScene().children, true);
    
    const hit = intersects.find(i => 
        i.object.userData?.isPhysicsBody && 
        !i.object.userData?.isStatic &&
        !i.object.userData?.ignoreRaycast
    );
    
    if (hit) {
        const physics = context.getAVBDPhysics();
        if (physics?.world.startDrag(hit.object.userData.physicsBodyId, hit.point.x, hit.point.y)) {
            isGrabbing = true;
            grabInputType = 'mouse';
            grabDistance = hit.distance;
            if (context.orbitControls) context.orbitControls.enabled = false;
            updateHighlight(hit.object);
        }
    }
}

function handleMouseMove(e) {
    if (!isGrabbing || grabInputType !== 'mouse') return;
    const ndc = getMouseNDC(e);
    const camera = context.getCamera();
    raycaster.setFromCamera(ndc, camera);
    const target = raycaster.ray.origin.add(raycaster.ray.direction.multiplyScalar(grabDistance));
    const physics = context.getAVBDPhysics();
    physics?.world.updateDrag(target.x, target.y);
    
    if (currentHighlight && currentHighlight.material) {
        currentHighlight.material.resolution.set(context.canvasWidth, context.canvasHeight);
    }
}

function handleMouseUp() {
    if (isGrabbing && grabInputType === 'mouse') {
        endDrag();
        clearHighlight();
    }
}

$effect(() => {
    const renderer = context.getRenderer();
    if (!renderer) return;
    const canvas = renderer.domElement;
    canvasElement = canvas;
    
    canvas.addEventListener('pointerdown', handleMouseDown);
    canvas.addEventListener('pointermove', handleMouseMove);
    canvas.addEventListener('pointerup', handleMouseUp);
    
    return () => {
        canvas.removeEventListener('pointerdown', handleMouseDown);
        canvas.removeEventListener('pointermove', handleMouseMove);
        canvas.removeEventListener('pointerup', handleMouseUp);
        if (canvasElement === canvas) canvasElement = null;
    };
});
</script>
<!--
/*
 * ASSESS/CONSIDER A DIFFERENT APPROACH:
 * ok. I want to invent a new METHOD to positively "grab" using the hand detection landmarks.
 * GOAL:
 * - When the CYAN landmarks overlap, that should count as a "grab".
 * - this should use the XY co-ords of camera projection pixels to acheive this.
**/
-->
