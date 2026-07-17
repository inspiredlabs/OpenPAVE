// ./main.js - Encapsulate all `requestAnimationFrame` step functions
import * as THREE from 'three';
import { useContext } from './simpleContext.svelte.js'; // Ensure path is correct

/**
 * Three.js setup that uses the shared context
 * @returns {import('svelte/attachments').Attachment}
 */
export function main(width, height) { // width/height params from App.svelte are initial values
  return (canvas) => {
    // console.log("Setting up Three.js with context (main.js)");
    
    const context = useContext();

    // Use dimensions from context first, then fall back to passed or window
    let currentWidth = context.canvasWidth || width || window.innerWidth;
    let currentHeight = context.canvasHeight || height || window.innerHeight;
    
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true }); // antialias: true is good
    renderer.setClearColor(0x000000, 0); // Enable transparency
    renderer.setSize(currentWidth, currentHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true; // Enable shadows globally
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    context.renderer = renderer; // Store renderer in context
    const scene = new THREE.Scene();
    scene.background = null;
    context.setScene(scene);
    
    // --- FPS Camera (Player's View) ---
    const fpsCamera = new THREE.PerspectiveCamera(60, currentWidth / currentHeight, 0.1, 5000);
    fpsCamera.position.set(0, 1.6, 9.8); // More standard FPS start: y=1.6 (eye height), z=10 (distance from origin)
    const initialFpsLookAt = new THREE.Vector3(0, 1.6, 0); // Look forward from initial position
    fpsCamera.lookAt(initialFpsLookAt);
    context.setFpsCamera(fpsCamera); // Register FPS camera with context

    // Set the initial generic LookAt for CameraControls UI to match FPS camera's target
    if (context.setLookAtTarget) {
        context.setLookAtTarget(initialFpsLookAt.x, initialFpsLookAt.y, initialFpsLookAt.z);
    }
    // Set initial shared Yaw/Pitch for CameraControls UI based on FPS camera's orientation
    if (context.setCameraYaw && context.setCameraPitch) {
        context.setCameraYaw(fpsCamera.rotation.y);
        context.setCameraPitch(fpsCamera.rotation.x);
    }

    // --- Drone Camera (Track-Side View) ---
    const droneCamera = new THREE.PerspectiveCamera(50, currentWidth / currentHeight, 0.1, 1000);
    // Where to look:
    const droneLookAtTargetPos = new THREE.Vector3(0, 7, -40); // Look at approx center of your sphere column
    droneCamera.position.set(80, 7, -40); // Positioned to the side and slightly elevated
    droneCamera.lookAt(droneLookAtTargetPos);
    context.setDroneCamera(droneCamera); // Register Drone camera with context
        
    // --- Set Initial Active Camera ---
    // Use setActiveCameraType as defined in your latest context
    if (context.setActiveCameraType) {
      context.setActiveCameraType('fps'); // Start 'fps' ONLY 'drone', not configured here
    } else {
      console.error("[main.js] context.setActiveCameraType is not defined!");
      // Fallback if method doesn't exist (older context version)
      // This part should ideally not be needed if context is up-to-date.
      if (context.setCamera) context.setCamera(fpsCamera);
    }
    
    // --- Lights ---
    scene.add(new THREE.AmbientLight(0xffffff, 1.0)); // Ambient light
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(15, 20, 10); // General lighting direction
    // directionalLight.castShadow = true;
    // directionalLight.shadow.mapSize.width = 1024; // Decent shadow map size
    // directionalLight.shadow.mapSize.height = 1024;
    // directionalLight.shadow.camera.far = 50; // Adjust shadow camera frustum
    // directionalLight.shadow.camera.left = -25;
    directionalLight.shadow.camera.right = 25;
    directionalLight.shadow.camera.top = 25;
    directionalLight.shadow.camera.bottom = -25;
    scene.add(directionalLight);
    
    // --- Objects --- 
		/* EXTREMIS BOTTOM OF GAME WORLD */
		/* `YellowSphere` should not intersect this mesh */
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(100, 200), // Larger plane
      new THREE.MeshStandardMaterial({ color: 0x777777, side: THREE.DoubleSide, roughness: 0.9 })
    );
    plane.rotation.x = -Math.PI / 2;
    plane.receiveShadow = true; // Ground should receive shadows
    scene.add(plane);

    // --- Objects --- 
		/* EXTREMIS TOP OF GAME WORLD */
		/* `YellowSphere` should not intersect this mesh */
		const ceiling = new THREE.Mesh(
      new THREE.PlaneGeometry(100, 200), // Larger plane
      new THREE.MeshStandardMaterial({
				color: 0x777777,
				emissiveIntensity: 0.5,
				side: THREE.DoubleSide,
				roughness: 0.9 })
    );
		ceiling.position.set(0, 13, 0);
    ceiling.rotation.x = -Math.PI / 2;
    // scene.add(ceiling);
		
    // Your example objects (ensure castShadow is true for those that should cast shadows)
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.0, 32, 16), 
      new THREE.MeshStandardMaterial({ color: 0x00ee00 }) // Brighter green
    );
    sphere.position.set(-5, 1, 0);
    sphere.castShadow = true;
    // scene.add(sphere);
    
    const cylinder = new THREE.Mesh(
      new THREE.CylinderGeometry(1, 1, 2, 32), // Smoother cylinder
      new THREE.MeshStandardMaterial({ color: 0xee0000 }) // Brighter red
    );
    cylinder.position.set(0, 1, 2); // Sitting on the plane
    cylinder.castShadow = true;
    // scene.add(cylinder);
    
    window.THREE = THREE; // For doomControls or global access if needed
        
    function checkResize() {
      const newWidth = context.canvasWidth;
      const newHeight = context.canvasHeight;
      const activeCam = context.getCamera(); // Get the currently active camera
      
      if (renderer && activeCam && newWidth && newHeight && 
          (newWidth !== currentWidth || newHeight !== currentHeight)) {
        currentWidth = newWidth;
        currentHeight = newHeight;
        
        renderer.setSize(newWidth, newHeight);
        
        activeCam.aspect = newWidth / newHeight;
        activeCam.updateProjectionMatrix();
      }
    }
    context.registerUpdatable(checkResize);
    
    let frameId;
    let lastFrameTime = 0;
    function animate(currentTime) {
      if (lastFrameTime === 0) lastFrameTime = currentTime;
      const deltaTime = (currentTime - lastFrameTime) / 1000;
      lastFrameTime = currentTime;

      const activeCam = context.getCamera();
      const frameStart = performance.now();
      let updatersMs = 0;
      let renderMs = 0;
      
      const updatersStart = performance.now();
      for (const updatable of context._getUpdatableFunctions()) {
        if (typeof updatable === 'function') {
          try {
            updatable(deltaTime, currentTime, activeCam);
          } catch (e) {
            console.error("Error in updatable function:", e);
          }
        }
      }
      updatersMs = performance.now() - updatersStart;

      // Add pixelatedRendering.js
      const renderStart = performance.now();
      const composer = context.getComposer?.();
      if (composer) {
        composer.render(); // Handles scene + effects in one pass
      } else if (activeCam && scene) {
        renderer.render(scene, activeCam); // Fallback for no effects
      }
      renderMs = performance.now() - renderStart;

      context.recordMainFramePerf?.({
        frameMs: performance.now() - frameStart,
        deltaMs: deltaTime * 1000,
        updatersMs,
        renderMs
      });
      
      frameId = requestAnimationFrame(animate);
    }
    frameId = requestAnimationFrame(animate);
    
    return () => { // Cleanup
      console.log("Cleaning up Three.js (main.js attachment)");
      if (frameId) cancelAnimationFrame(frameId);
      
      if (context.unregisterUpdatable) { // Check if method exists before calling
          context.unregisterUpdatable(checkResize);
      }
      
      scene.traverse(object => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(material => {
                if(material.map) material.map.dispose(); // Dispose textures
                material.dispose();
            });
          } else {
            if(object.material.map) object.material.map.dispose(); // Dispose textures
            object.material.dispose();
          }
        }
      });
      if (renderer) renderer.dispose();
      
      // Clear specific cameras from context
      if (context.setFpsCamera) context.setFpsCamera(null);
      if (context.setDroneCamera) context.setDroneCamera(null);
      // Reset active camera type or clear active camera in context if desired
      // if (context.setActiveCameraType) context.setActiveCameraType('fps'); // Or some default
      
      // Nullify general references if they are part of the context's API
      if (context.setScene) context.setScene(null);
      // context.camera = null; // Avoid direct assignment if using setActiveCameraType
      if (context.setRenderer) context.setRenderer(null);

      console.log("Three.js cleanup complete.");
    };
  };
}
