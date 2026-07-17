<script>
// ./OrbitControlsComp.svelte
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useContext } from './simpleContext.svelte.js';

const context = useContext();

$effect(() => {
  const camera = context.getCamera();
  const renderer = context.renderer;
  if (!camera || !renderer) return;

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // XPBD physics body picker store in context for PickingControls:
  context.orbitControls = controls;

  const updateFn = (delta) => {
    const activeCamera = context.getCamera();
    const isFpsCamera = context.getActiveCameraType?.() === 'fps';

    if (activeCamera && controls.object !== activeCamera) {
      controls.object = activeCamera;
    }

    controls.enabled = !isFpsCamera;
    if (isFpsCamera) return;

    controls.update();
  };

  context.registerUpdatable(updateFn);

  return () => {
    controls.dispose();
    context.orbitControls = null;
    context.unregisterUpdatable(updateFn);
  };
});
</script>
