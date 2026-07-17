<script>
// +page.svelte - Mediapipe Hands Demo
import { main } from "./main.js";
import { createContext } from './simpleContext.svelte.js';
import { doomControls } from './doomControls.js';

// Three.js components
// import OrbitControlsComp from "./OrbitControlsComp.svelte";
import PickingControls from "./PickingControls.svelte";

// Declarative game objects:
import FpsCounter from './FpsCounter.svelte';
// import GizmoControls from './GizmoControls.svelte';
import CopyJSON from './CopyJSON.svelte'; // Monitor sceneGraph
import HandLandmarkChart from './HandLandmarkChart.svelte';
// import CameraControls from './CameraControls.svelte';

/*
import YellowSphere from './YellowSphere.svelte';	
import FloatingBoxes from './FloatingBoxes.svelte'; 
import BoxingGlove from './BoxingGlove.svelte';
*/	

// Unified pose-lite + hands pipeline
import { mediapipeVisionPoseLiteHands } from './mediapipeVisionPoseLiteHands.js';


// Keep ONE context: the core feature of Svelte's context system
// Create context at the app level, and reference it too
// `createContext()` returns it to you, so you don't
// need to immediately call `useContext()`.
const context = createContext();


// ============================================
// SCALE BROWSER VIEWPORT
// ============================================
let width = $state(window.innerWidth);
let height = $state(window.innerHeight);

$effect(() => {
	if (context) {
		context.canvasWidth = width;
		context.canvasHeight = height;
	}
});

function handleResize() {
	width = window.innerWidth;
	height = window.innerHeight;
}

$effect(() => {
	handleResize();
	window.addEventListener("resize", handleResize);
	return () => window.removeEventListener("resize", handleResize);
});

function init(canvas) {
	console.log("[main.js] mount ThreeJS canvas");
	const cleanup = main()(canvas);
	return () => {
		if (cleanup) cleanup();
	};
}


// ============================================
// DEBUG STATE
// ============================================
let debugUI = $state(context.debugUI);

$effect(() => {
	const sync = () => (debugUI = context.debugUI);
	context.subscribe(sync);
	return () => context.unsubscribe(sync);
});

function handleKeydown(event) {
	if (event.key.toLowerCase() === "escape") {
		context.debugUI = !debugUI;
		event.preventDefault();
	}
}


// ============================================
// MEDIAPIPE: gesture controller & screen ratio
// ============================================
// Screen Config
let targetPlaneZ = $state(0.0); // Anchor point for aiming system!
let viewportScale = $state(0.2);
let videoOpacity = $state(0.0);
let previewOpacity = $state(1.0);
// Smooth object pool meshes
let drawSkeletons = $state(true); // false
let filterAlpha = $state(0.25); // Low-pass filter (smooth algo).
// Provide interactivity:
let enableHeadOrientationControl = $state(false); // false, before game start
let maxYawDegrees = $state(-45); // min 0 avg. 5 max 60 @edges
let maxPitchDegrees = $state(15); // min 0 avg. 30 max 90 @floor/sky
let headOrientationSmoothing = $state(0.2);
let autoForwardInterval = $state(null);
let drawPoseOverlay = $state(false);
let drawHandPreview = $state(true);
let optimizedMediaPipe = $state(true);
let mediaPipeProcessWidth = $state(640);
let poseTargetFps = $state(15);
let handTargetFps = $state(60);
let enableSceneDeltaSkip = $state(false);
let sceneDeltaThreshold = $state(1.5); // pose if camera image change
let forcePoseRefreshMs = $state(500); // pose every 1/2 second

async function mediapipePerfTrace() {
	const summary = context.mediapipePerfTrace?.({ durationMs: 10_000, log: true });
	if (!summary) {
		console.warn('[MP Perf Trace] No MediaPipe trace data yet.');
		return;
	}

	try {
		await navigator.clipboard.writeText(JSON.stringify(summary, null, 2));
		console.log('[MP Perf Trace] Copied summary JSON to clipboard.');
	} catch (error) {
		console.warn('[MP Perf Trace] Could not copy summary JSON to clipboard:', error);
	}
}


</script>

<!-- BELOW: -->
<!-- Three.js canvas with improved attachments -->
<canvas
{@attach init}
{@attach doomControls()}
>
<!-- This space has been intentionally left blank -->
</canvas>

<!-- ABOVE: -->
<div class="selfie" {@attach mediapipeVisionPoseLiteHands({
drawSkeletons,
videoOpacity,
previewOpacity,
drawPoseOverlay,
drawHandPreview,
filterAlpha,
viewportScale,
enableHeadOrientationControl,
maxYawDegrees,
maxPitchDegrees,
orientationFilterAlpha: headOrientationSmoothing,
enablePoseDetection: false,
enableHandDetection: true,
maxHands: 2,
optimized: optimizedMediaPipe,
processWidth: mediaPipeProcessWidth,
poseTargetFps,
handTargetFps,
enableSceneDeltaSkip,
sceneDeltaThreshold,
forcePoseRefreshMs,
delegate: 'AUTO'
})}></div>

<FpsCounter />
<!-- <OrbitControlsComp /> -->
<PickingControls />
<CopyJSON />
<HandLandmarkChart staleAfterMs={750} />

<!-- THIS DOES WORK! -->
<!-- <YellowSphere 
radius={0.33}
distanceInFrontOfCamera={5.2}
showBoundingSphere={true}
/> -->
<!-- THIS DOES WORK! -->
<!-- <FloatingBoxes boxGridZ={1.8} /> -->

<!-- Show content when scene is ready
{#if debugUI}
	<BoxingGlove 
		hand="left" 
		color={0xff0000} 
		visible={true} 
		scale={2.5}
	/>
	<BoxingGlove 
		hand="right" 
		color={0x0000ff} 
		visible={true} 
		scale={2.5}
	/>
{/if}
-->



<div class="features">
<button style="display:block" onclick={() => context.getAVBDPhysics()?.inspectGPU()}>
	🔍 Inspect Adapter
</button>
<button style="display:block" onclick={mediapipePerfTrace}>
	👍 MP Perf. Trace
</button>
</div>



<svelte:window
	bind:innerHeight={height}
	bind:innerWidth={width}
	on:keydown={handleKeydown}
/>

<style>
:global(body) {
	margin: 0;
	overflow: hidden;
	background: goldenrod;
	font-family: sans-serif;
}

canvas {
	/* pointer-events: auto; */
	/* pointer-events: none; */
	display: block;
	position: absolute;
	top: 0; left: 0;
	width: 100%; height: 100%;
	outline: none;
	z-index: 3
}
:global(small) {
	position: absolute;
	right: 20px;
	background: rgba(0, 0, 0, 0.3);
	color: lime;
	padding: 0 15px;
	border-radius: 8px;
	font-family: monospace;
	font-size: 0.7em;
	z-index: 1;
	width: 108px;
	min-height: 5em;
	overflow-y: scroll;
	overflow-x: hidden;
	z-index: 9
}
:global(small input) {
	width: 100%
}

.features {
	position: fixed;
	top: 20px;
	right: 20px;
	z-index: 999;
	padding: 0.25rem 0.5rem;
}
</style>

<!-- 
/* PREVENT CLUTTER */
	
:global(*){
box-sizing: border-box;

/* disable text selection (svg icons are also text...) */
-webkit-user-select: none;
/* Safari */
-ms-user-select: none;
/* IE 10 and IE 11 */
user-select: none;
/* Standard syntax */

/* preventing the long press context menu, https://stackoverflow.com/a/56866766/3022127 */
-webkit-touch-callout: none !important;
-webkit-user-select: none !important;

/* preventing iOS tap highlight */
-webkit-tap-highlight-color: transparent;

/* Disable browser handling of all panning and zooming gestures, except for regular scrolling */
touch-action: pan-y;
} -->
