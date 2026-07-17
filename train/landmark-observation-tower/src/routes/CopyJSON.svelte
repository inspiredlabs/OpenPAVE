<script>
  // ./CopyJSON.svelte - CORRECTED & SIMPLIFIED
  import { useContext } from './simpleContext.svelte.js';
	
  const context = useContext();
  let textarea;

  // Initial content for the text area
  let jsonContent = $state('Click "Refresh JSON" to see the scene graph.');

  // --- Scene Graph Serialization Function ---
  // This function converts a Three.js object into a plain object for JSON.
  function serializeObject(obj) {
    if (!obj) return null;

    const serialized = {
      name: obj.name || '(unnamed)',
      type: obj.type,
      uuid: obj.uuid,
      visible: obj.visible,
      // Use .toFixed(3) to keep the JSON readable
      position: { 
        x: parseFloat(obj.position.x.toFixed(3)),
        y: parseFloat(obj.position.y.toFixed(3)),
        z: parseFloat(obj.position.z.toFixed(3))
      },
      // You can add rotation and scale if needed
      // rotation: { x: obj.rotation.x, y: obj.rotation.y, z: obj.rotation.z },
      // scale: { x: obj.scale.x, y: obj.scale.y, z: obj.scale.z },
      children: []
    };

    if (obj.children && obj.children.length > 0) {
      // Recursively serialize children
      serialized.children = obj.children.map(child => serializeObject(child));
    }

    return serialized;
  }

  // --- Manual Refresh Function ---
  function refreshJson() {
    if (!context || !context.getScene()) {
      jsonContent = JSON.stringify({ error: "Context or Scene not ready." }, null, 2);
      return;
    }

    // Get the full scene graph at the moment the button is clicked
    const sceneGraph = serializeObject(context.getScene());
    
    // Combine it with other useful debug info
    const debugData = {
      sceneGraph,
      activeCameraType: context.getActiveCameraType(),
      // mediaPipeInitialized: context.getMediaPipeInitialized(),
      // faceLandmarksCount: context.getFaceLandmarks()?.[0]?.length || 0,
      // hasTransformMatrix: !!context.getFacialTransformationMatrix(),
    };

    // Update the state variable with the new JSON string
    jsonContent = JSON.stringify(debugData, null, 2);
  }

  // --- Copy to Clipboard Logic ---
  let copyStatus = $state('');
  function copy() {
    if (!textarea) return;
    try {
      navigator.clipboard.writeText(textarea.value).then(() => {
        copyStatus = 'Copied';
        setTimeout(() => { copyStatus = ''; }, 2000);
      });
    } catch (err) {
      copyStatus = 'Error';
    }
  }


// Escape Key	UI:
let debugUI = $state(context.debugUI);

$effect(() => {
	const sync = () => debugUI = context.debugUI;
	context.subscribe(sync);
	return () => context.unsubscribe(sync);
});
</script>

<!-- {#if debugUI} -->
<aside>
  <textarea bind:this={textarea} readonly>{jsonContent}</textarea>
  <!-- Buttons for manual control -->
  <button onclick={refreshJson}>Refresh</button>
	<button onclick={copy}>{copyStatus || 'Copy'}</button>
</aside>
<!-- {/if} -->

<style>
  aside {
    position: absolute;
    top: 40px;
    left: 10px;
    font-family: monospace;
    z-index: 100;
    width: 128px;
    overflow-x: hidden;
    overflow-y: auto !important;
  }
</style>