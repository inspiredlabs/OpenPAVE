// ./doomControls.js
import * as THREE from 'three';
import { useContext } from './simpleContext.svelte.js'; // Ensure path is correct

export function doomControls() {
  return (canvas) => { // canvas is the DOM element this attachment is applied to
    const context = useContext();
    
    // Movement state
    const moveState = { 
      forward: false,
			backward: false,
			// left: false,
			// right: false, 
      up: false,
			down: false 
    };
    const velocity = new THREE.Vector3();
    const speed = 0.07; // Slightly increased speed for better feel
    const mouseSensitivity = 0.002;

    // No local yaw/pitch for doomControls itself; it uses context's yaw/pitch for FPS mode
    
    function handleKeyDown(event) {
      // Camera Switching Logic
      if (event.key === '0') {
        if (context.setActiveCameraType) {
          context.setActiveCameraType('fps');
        }
        return; // Consume the event if it's for camera switching
      } else if (event.key === '9') {
        if (context.setActiveCameraType) {
          context.setActiveCameraType('drone');
        }
        return; // Consume the event
      }

      // Only process movement keys if FPS camera is active
      if (context.getActiveCameraType && context.getActiveCameraType() !== 'fps') {
        return;
      }

      switch (event.code) {
        case 'KeyW': case 'ArrowUp': moveState.forward = true; break;
        case 'KeyS': case 'ArrowDown': moveState.backward = true; break;
        // case 'KeyA': case 'ArrowLeft': moveState.left = true; break;
        // case 'KeyD': case 'ArrowRight': moveState.right = true; break;
        case 'Space': moveState.up = true; break;
        case 'ShiftLeft': case 'ControlLeft': moveState.down = true; break; // Use Control as well
      }
    }
    
    function handleKeyUp(event) {
      // No need to check active camera for keyup, just reset flags
      switch (event.code) {
        case 'KeyW': case 'ArrowUp': moveState.forward = false; break;
        case 'KeyS': case 'ArrowDown': moveState.backward = false; break;
        // case 'KeyA': case 'ArrowLeft': moveState.left = false; break;
        // case 'KeyD': case 'ArrowRight': moveState.right = false; break;
        case 'Space': moveState.up = false; break;
        case 'ShiftLeft': case 'ControlLeft': moveState.down = false; break;
      }
    }
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    
    function handleMouseMove(event) {
      // Only process mouse look if FPS camera is active and left button is down
      if (context.getActiveCameraType && context.getActiveCameraType() !== 'fps') {
        return;
      }
      if (event.buttons !== 1 && document.pointerLockElement !== canvas) return; // Require left click or pointer lock

      let currentYaw = context.getCameraYaw ? context.getCameraYaw() : 0;
      let currentPitch = context.getCameraPitch ? context.getCameraPitch() : 0;

      currentYaw -= event.movementX * mouseSensitivity;
      currentPitch -= event.movementY * mouseSensitivity;
      // Clamping is handled by context.setCameraPitch

      if (context.setCameraYaw) context.setCameraYaw(currentYaw);
      if (context.setCameraPitch) context.setCameraPitch(currentPitch);
    }
    
    // Pointer lock for better mouse control in FPS mode
    function requestPointerLock() {
        if (context.getActiveCameraType && context.getActiveCameraType() === 'fps') {
            canvas.requestPointerLock();
        }
    }

    function syncHeadOrientationWithPointerLock() {
      const isPointerLocked = document.pointerLockElement === canvas;
      context.setHeadOrientationControlEnabled?.(!isPointerLocked);
    }

    canvas.addEventListener('click', requestPointerLock); // Lock pointer on click when FPS is active
    canvas.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('pointerlockchange', syncHeadOrientationWithPointerLock);
    
    // Updatable function (called every frame by main.js animation loop)
    function updateDoomControls(deltaTime, currentTime, activeCamera) { // activeCamera passed from main.js
      // This function only operates if the FPS camera is the active one
      if (!activeCamera || !context.getActiveCameraType || context.getActiveCameraType() !== 'fps' || activeCamera !== context.getFpsCamera()) {
        return;
      }

      // The 'activeCamera' passed in should be the fpsCamera if logic above is met
      const fpsCam = activeCamera;

      context.applyHeadOrientationSmoothing?.(deltaTime);

      // Get current shared yaw and pitch from context
      const yaw = context.getCameraYaw ? context.getCameraYaw() : 0;
      const pitch = context.getCameraPitch ? context.getCameraPitch() : 0;

      // Reset velocity
      velocity.set(0, 0, 0);

      // Calculate movement direction based on input state
      if (moveState.forward) velocity.z -= 1;
      if (moveState.backward) velocity.z += 1;
      // if (moveState.left) velocity.x -= 1;
      // if (moveState.right) velocity.x += 1;

      // Normalize diagonal movement and apply speed
      if (velocity.lengthSq() > 0) { // Check if there's any XZ movement
          velocity.normalize().multiplyScalar(speed);
      }
      
      // Vertical movement (local Y for FPS camera)
      let verticalVelocity = 0;
      if (moveState.up) verticalVelocity += speed;
      if (moveState.down) verticalVelocity -= speed;

      // Create a quaternion for the yaw rotation (around world Y axis)
      const yawQuaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
      
      // Apply yaw to the XZ movement vector
      velocity.applyQuaternion(yawQuaternion);
      
      // Add calculated XZ movement to camera's current position
      fpsCam.position.add(velocity);
      
      // Add vertical movement directly to camera's Y position
      fpsCam.position.y += verticalVelocity;
      
      // Apply pitch and yaw for camera orientation
      // Using Euler angles with 'YXZ' order is typical for FPS controls.
      // Yaw is applied first (around Y), then Pitch (around the new X).
      fpsCam.rotation.set(pitch, yaw, 0, 'YXZ');
    }
    
    // Poll for context readiness to register the updatable function
    let updatableRegistered = false;
    let intervalId = null;
    
    function tryRegisterUpdater() {
      // Ensure context and its methods are available
      if (context && context.registerUpdatable && context.getActiveCameraType) {
        context.registerUpdatable(updateDoomControls);
        updatableRegistered = true;
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        // console.log("[DoomControls] Controls updater registered.");
        return true;
      }
      return false;
    }
    
    if (!tryRegisterUpdater()) {
      // console.log("[DoomControls] Context not fully ready, will poll to register updater.");
      intervalId = setInterval(tryRegisterUpdater, 100);
    }
    
    // Cleanup when the attachment is destroyed
    return () => {
      console.log("Cleaning up doom controls");
      
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('click', requestPointerLock);
      document.removeEventListener('pointerlockchange', syncHeadOrientationWithPointerLock);
      
      // Exit pointer lock if active
      if (document.pointerLockElement === canvas) {
        document.exitPointerLock();
      }

      context.setHeadOrientationControlEnabled?.(true);
      
      if (intervalId) {
        clearInterval(intervalId);
      }
      
      if (updatableRegistered && context && context.unregisterUpdatable) {
        context.unregisterUpdatable(updateDoomControls);
      }
    };
  };
}
