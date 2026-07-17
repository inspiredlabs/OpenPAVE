<script>
  import { untrack } from 'svelte';
  import { useContext } from './simpleContext.svelte.js';

  let { staleAfterMs = 750, maxRecordedFrames = 1200 } = $props();

  const context = useContext();
  let result = $state.raw(null);
  let paused = $state(false);
  let collapsed = $state(false);
  let copyStatus = $state('');
  let recording = $state(false);
  let recordedFrameCount = $state(0);
  let recordedHandCount = $state(0);
  let rejectedFrameCount = $state(0);
  let lastSequence = -1;
  let lastReceivedSequence = -1;
  let pendingFrame = null;
  let copyStatusTimeout = null;
  let recordedFrames = [];
  let recordingStartedAt = null;

  let handCount = $derived(result?.landmarks?.length ?? 0);

  function receiveFrame(frame) {
    if (!frame) {
      pendingFrame = null;
      result = null;
      return;
    }

    if (frame.sequence <= lastReceivedSequence) return;
    lastReceivedSequence = frame.sequence;

    if (recording) recordFrame(frame);
    if (paused) {
      // Pausing retains at most one pending reference, never a frame queue.
      pendingFrame = frame;
      return;
    }

    lastSequence = frame.sequence;
    pendingFrame = null;
    result = frame;
  }

  function recordFrame(frame) {
    const exportableHands = frame.oracleHands?.filter((hand) => hand.validity.exportable) ?? [];
    if (exportableHands.length === 0) {
      rejectedFrameCount += 1;
      return;
    }
    if (recordedFrames.length >= maxRecordedFrames) {
      recording = false;
      return;
    }

    // Frames are immutable-by-contract and already own copied MediaPipe data.
    // Store one reference; do not clone or create a second history.
    recordedFrames.push(frame);
    recordedFrameCount = recordedFrames.length;
    recordedHandCount += exportableHands.length;
    if (recordedFrames.length >= maxRecordedFrames) recording = false;
  }

  function clearRecording() {
    recording = false;
    recordedFrames = [];
    recordedFrameCount = 0;
    recordedHandCount = 0;
    rejectedFrameCount = 0;
    recordingStartedAt = null;
  }

  function toggleRecording() {
    if (!recording && recordedFrames.length >= maxRecordedFrames) clearRecording();
    if (!recording && recordedFrames.length === 0) recordingStartedAt = new Date().toISOString();
    recording = !recording;
  }

  function exportBenchmark() {
    if (recordedFrames.length === 0) return;
    const dataset = {
      schema: 'openpave.oracle-roi.dataset.v1',
      createdAt: recordingStartedAt,
      exportedAt: new Date().toISOString(),
      purpose: 'oracle_roi_landmarker_benchmark',
      teacher: 'mediapipe_hand_landmarker',
      supervision: {
        normalizedLandmarks: 'teacher_pseudo_label',
        worldLandmarks: 'teacher_pseudo_label',
        palmAndSkeletonNormals: 'derived_teacher_pseudo_label',
        skinSurfaceNormals: 'unavailable',
        accurateNormalReferenceRequired: 'synthetic_rig_or_rgbd'
      },
      selection: {
        oracleRoi: true,
        acquisitionModelBypassed: true,
        onlyFramesWithExportableHands: true,
        maxRecordedFrames
      },
      counters: {
        frames: recordedFrames.length,
        exportableHands: recordedHandCount,
        rejectedFrames: rejectedFrameCount
      },
      frames: recordedFrames
    };
    const blob = new Blob([JSON.stringify(dataset)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `openpave-oracle-roi-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function togglePaused() {
    paused = !paused;
    if (!paused && pendingFrame) receiveFrame(pendingFrame);
  }

  function formatNumber(value) {
    if (!Number.isFinite(value)) return '—';
    if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(2);
    return value.toFixed(6);
  }

  async function copyResult() {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      copyStatus = 'Copied';
    } catch {
      copyStatus = 'Error';
    }
    if (copyStatusTimeout) clearTimeout(copyStatusTimeout);
    copyStatusTimeout = setTimeout(() => {
      copyStatus = '';
      copyStatusTimeout = null;
    }, 1500);
  }

  $effect(() => {
    const unsubscribe = context.subscribeHandFrames((frame) =>
      untrack(() => receiveFrame(frame))
    );
    const staleInterval = setInterval(() => {
      if (!paused && result && performance.now() - result.capturedAt > staleAfterMs) {
        result = null;
      }
    }, Math.min(250, Math.max(50, staleAfterMs / 2)));

    return () => {
      unsubscribe();
      clearInterval(staleInterval);
      if (copyStatusTimeout) clearTimeout(copyStatusTimeout);
      pendingFrame = null;
      result = null;
      recordedFrames = [];
    };
  });
</script>

<aside class:collapsed aria-label="Hand landmark stream">
  <header>
    <button
      class="collapse"
      title={collapsed ? 'Expand landmark stream' : 'Collapse landmark stream'}
      onclick={() => (collapsed = !collapsed)}
    >{collapsed ? '▸' : '▾'}</button>
    <strong>Oracle ROI benchmark</strong>
    <span class:live={!paused}>{paused ? 'paused' : 'live'}</span>
  </header>

  {#if !collapsed}
    <div class="toolbar">
      <span>{handCount} {handCount === 1 ? 'hand' : 'hands'}</span>
      <span>{result ? `frame ${result.sequence}` : 'no current frame'}</span>
      <button onclick={togglePaused}>{paused ? 'Resume' : 'Pause'}</button>
      <button disabled={!result} onclick={copyResult}>{copyStatus || 'Copy'}</button>
    </div>

    <div class="toolbar recorder">
      <span>{recordedFrameCount}/{maxRecordedFrames} frames · {recordedHandCount} hands</span>
      <button class:recording onclick={toggleRecording}>{recording ? 'Stop' : 'Record'}</button>
      <button disabled={recordedFrameCount === 0} onclick={exportBenchmark}>Export</button>
      <button disabled={recordedFrameCount === 0} onclick={clearRecording}>Clear</button>
    </div>

    <div class="stream">
      {#if handCount === 0}
        <p class="empty">Waiting for a hand…</p>
      {:else}
        {#each result.landmarks as landmarks, handIndex}
          {@const categories = result.handedness?.[handIndex] ?? []}
          {@const worldLandmarks = result.worldLandmarks?.[handIndex] ?? []}
          {@const oracle = result.oracleHands?.[handIndex]}
          <section>
            <h2>Hand #{handIndex}</h2>

            <h3>Oracle ROI</h3>
            <dl>
              <dt>valid</dt><dd>{oracle?.oracleRoi?.valid ?? false}</dd>
              <dt>center</dt><dd>{oracle?.oracleRoi?.center?.map(formatNumber).join(', ') ?? '—'}</dd>
              <dt>size</dt><dd>{formatNumber(oracle?.oracleRoi?.size)}</dd>
              <dt>rotation</dt><dd>{formatNumber(oracle?.oracleRoi?.rotationRadians)}</dd>
            </dl>

            <h3>Palm frame / normal</h3>
            <dl>
              <dt>valid</dt><dd>{oracle?.palmFrame?.valid ?? false}</dd>
              <dt>normal</dt><dd>{oracle?.palmFrame?.axes?.z?.map(formatNumber).join(', ') ?? '—'}</dd>
              <dt>plane residual</dt><dd>{formatNumber(oracle?.palmFrame?.normalizedPlaneResidual)}</dd>
              <dt>bone frames</dt><dd>{oracle?.validity?.validBoneFrames ?? 0}/21</dd>
              <dt>bending normals</dt><dd>{oracle?.validity?.validBendingNormals ?? 0}/15</dd>
              <dt>exportable</dt><dd>{oracle?.validity?.exportable ?? false}</dd>
            </dl>

            <h3>Handedness</h3>
            {#each categories as category, categoryIndex}
              <div class="category">
                <h4>Category #{categoryIndex}</h4>
                <dl>
                  <dt>index</dt><dd>{category.index ?? '—'}</dd>
                  <dt>score</dt><dd>{formatNumber(category.score)}</dd>
                  <dt>categoryName</dt><dd>{category.categoryName || '—'}</dd>
                </dl>
              </div>
            {/each}

            <h3>Landmarks</h3>
            <table>
              <thead>
                <tr><th>#</th><th>x</th><th>y</th><th>z</th></tr>
              </thead>
              <tbody>
                {#each landmarks as landmark, landmarkIndex}
                  <tr>
                    <th>{landmarkIndex}</th>
                    <td>{formatNumber(landmark.x)}</td>
                    <td>{formatNumber(landmark.y)}</td>
                    <td>{formatNumber(landmark.z)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>

            <h3>WorldLandmarks</h3>
            {#if worldLandmarks.length > 0}
              <table>
                <thead>
                  <tr><th>#</th><th>x</th><th>y</th><th>z</th></tr>
                </thead>
                <tbody>
                  {#each worldLandmarks as landmark, landmarkIndex}
                    <tr>
                      <th>{landmarkIndex}</th>
                      <td>{formatNumber(landmark.x)}</td>
                      <td>{formatNumber(landmark.y)}</td>
                      <td>{formatNumber(landmark.z)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            {:else}
              <p class="empty compact">No world landmarks in this frame.</p>
            {/if}
          </section>
        {/each}
      {/if}
    </div>
  {/if}
</aside>

<style>
  aside {
    position: absolute;
    top: 40px;
    left: 150px;
    z-index: 100;
    width: min(430px, calc(100vw - 170px));
    max-height: calc(100vh - 55px);
    color: #dceeff;
    background: rgba(38, 38, 38, 0.8);
    overflow: hidden;
  }

  aside.collapsed { width: auto }

  .toolbar {
    display: flex;
    gap: 0.35rem;
    align-items: center;
  }

  .toolbar span { flex: 1 }

  button.recording {
    color: white;
    background: #a32626;
  }
</style>
