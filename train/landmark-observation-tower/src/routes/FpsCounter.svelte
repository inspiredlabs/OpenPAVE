<script>
	// ./FpsCounter.svelte
  let fps = 0;
  let lastTime = performance.now();
  let frameCount = 0;
  
  // Calculate FPS
  function updateFPS() {
    const currentTime = performance.now();
    frameCount++;
    
    // Calculate FPS every second
    if (currentTime - lastTime >= 1000) {
      fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
      frameCount = 0;
      lastTime = currentTime;
    }
    
    requestAnimationFrame(updateFPS);
  }
  
  // Start the FPS counter
  updateFPS();
</script>

<code class="pa3">
  FPS: {fps}
</code>

<style>
  code {
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 9999;
    background: rgba(0, 0, 0, 0.7);
    color: white;
    border-radius: 3px;
  }
</style>