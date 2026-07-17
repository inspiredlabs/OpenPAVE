<script>
// @ts-nocheck
import * as THREE from 'three';
import { useContext } from './simpleContext.svelte.js';

let {
  visible = true,
  viewportScale = 8.0,
  filterAlpha = 0.8,
  distanceInFrontOfCamera = 4.8,
  jointRadius = 0.055,
  boneRadius = 0.025,
  handColor = 0x35d9ff,
  palmColor = 0x1689b2,
  palmOpacity = 0.78
} = $props();

const context = useContext();
const LANDMARK_COUNT = 21;

// MediaPipe Hands topology. The palm perimeter is closed so it reads as a hand,
// while each finger is a separate articulated chain.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [17, 0]
];

// Five triangles form a broad, non-planar palm from the wrist to the knuckles.
const PALM_TRIANGLES = [
  [0, 1, 5],
  [1, 2, 5],
  [0, 5, 9],
  [0, 9, 13],
  [0, 13, 17]
];

let handGroup;
let landmarkMeshes = [];
let boneMeshes = [];
let palmMesh;
let palmPositionAttribute;
let isInitialized = false;

const filteredPositions = new Array(LANDMARK_COUNT)
  .fill(null)
  .map(() => new THREE.Vector3());
const positionInitialized = new Array(LANDMARK_COUNT).fill(false);

const cameraWorldPosition = new THREE.Vector3();
const cameraDirection = new THREE.Vector3();
const desiredGroupWorldPosition = new THREE.Vector3();
const boneDirection = new THREE.Vector3();
const boneMidpoint = new THREE.Vector3();
const cylinderAxis = new THREE.Vector3(0, 1, 0);
const mappedPosition = new THREE.Vector3();

function mapLandmarkToLocalPosition(landmark, target) {
  target.set(
    -(landmark.x - 0.5) * viewportScale,
    -(landmark.y - 0.5) * viewportScale,
    landmark.z * viewportScale
  );
}

function applyLowPassFilter(next, previous, alpha) {
  if (alpha >= 0.99) return next;
  const mappedAlpha = alpha * alpha;
  return mappedAlpha * next + (1 - mappedAlpha) * previous;
}

function updateBone(mesh, start, end) {
  boneDirection.subVectors(end, start);
  const length = boneDirection.length();

  if (length < 0.0001) {
    mesh.visible = false;
    return;
  }

  mesh.visible = true;
  boneMidpoint.addVectors(start, end).multiplyScalar(0.5);
  mesh.position.copy(boneMidpoint);
  mesh.quaternion.setFromUnitVectors(cylinderAxis, boneDirection.normalize());
  mesh.scale.set(1, length, 1);
}

function updatePalm() {
  let vertex = 0;
  for (const triangle of PALM_TRIANGLES) {
    for (const landmarkIndex of triangle) {
      const position = filteredPositions[landmarkIndex];
      palmPositionAttribute.setXYZ(vertex, position.x, position.y, position.z);
      vertex += 1;
    }
  }
  palmPositionAttribute.needsUpdate = true;
  palmMesh.geometry.computeVertexNormals();
}

function hideHand() {
  if (handGroup?.visible) handGroup.visible = false;
  positionInitialized.fill(false);
}

function perFrameUpdate() {
  if (!isInitialized || !handGroup) return;

  const camera = context.getCamera();
  const landmarks = context.getHandLandmarks();

  if (!visible || !camera || !landmarks || landmarks.length < LANDMARK_COUNT) {
    hideHand();
    return;
  }

  handGroup.visible = true;

  camera.getWorldPosition(cameraWorldPosition);
  camera.getWorldDirection(cameraDirection);
  desiredGroupWorldPosition
    .copy(cameraWorldPosition)
    .add(cameraDirection.multiplyScalar(distanceInFrontOfCamera));

  // Keep normalized webcam x/y aligned with the camera even while it rotates.
  handGroup.position.copy(desiredGroupWorldPosition);
  camera.getWorldQuaternion(handGroup.quaternion);

  for (let i = 0; i < LANDMARK_COUNT; i++) {
    mapLandmarkToLocalPosition(landmarks[i], mappedPosition);

    if (!positionInitialized[i]) {
      filteredPositions[i].copy(mappedPosition);
      positionInitialized[i] = true;
    } else {
      const current = filteredPositions[i];
      current.set(
        applyLowPassFilter(mappedPosition.x, current.x, filterAlpha),
        applyLowPassFilter(mappedPosition.y, current.y, filterAlpha),
        applyLowPassFilter(mappedPosition.z, current.z, filterAlpha)
      );
    }

    landmarkMeshes[i].position.copy(filteredPositions[i]);
  }

  for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
    const [startIndex, endIndex] = HAND_CONNECTIONS[i];
    updateBone(boneMeshes[i], filteredPositions[startIndex], filteredPositions[endIndex]);
  }

  updatePalm();
}

$effect(() => {
  const scene = context.getScene();
  if (!scene) return;

  handGroup = new THREE.Group();
  handGroup.name = 'mediaPipeHandSkeleton';
  handGroup.visible = false;
  scene.add(handGroup);

  const jointGeometry = new THREE.SphereGeometry(jointRadius, 12, 8);
  const boneGeometry = new THREE.CylinderGeometry(boneRadius, boneRadius, 1, 8);
  const jointMaterial = new THREE.MeshBasicMaterial({ color: handColor });
  const boneMaterial = new THREE.MeshBasicMaterial({ color: handColor });
  const palmMaterial = new THREE.MeshBasicMaterial({
    color: palmColor,
    transparent: palmOpacity < 1,
    opacity: palmOpacity,
    side: THREE.DoubleSide,
    depthWrite: palmOpacity >= 1
  });

  for (let i = 0; i < LANDMARK_COUNT; i++) {
    const joint = new THREE.Mesh(jointGeometry, jointMaterial);
    joint.name = `handLandmark-${i}`;
    handGroup.add(joint);
    landmarkMeshes.push(joint);
  }

  for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
    const bone = new THREE.Mesh(boneGeometry, boneMaterial);
    bone.name = `handBone-${HAND_CONNECTIONS[i][0]}-${HAND_CONNECTIONS[i][1]}`;
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

  palmMesh = new THREE.Mesh(palmGeometry, palmMaterial);
  palmMesh.name = 'handPalm';
  // Draw the translucent surface just behind the joints and bones.
  palmMesh.renderOrder = -1;
  palmMesh.frustumCulled = false;
  handGroup.add(palmMesh);

  isInitialized = true;
  context.registerUpdatable(perFrameUpdate);

  return () => {
    isInitialized = false;
    context.unregisterUpdatable(perFrameUpdate);
    scene.remove(handGroup);

    jointGeometry.dispose();
    boneGeometry.dispose();
    palmGeometry.dispose();
    jointMaterial.dispose();
    boneMaterial.dispose();
    palmMaterial.dispose();

    landmarkMeshes = [];
    boneMeshes = [];
    palmMesh = null;
    palmPositionAttribute = null;
    handGroup = null;
  };
});
</script>
