import { AlertTriangle, CircleStop, Play, RefreshCw } from 'lucide-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';

// App configuration ---------------------------------------------------------
// The foreground can run even when the C++ API is offline, but inference needs
// this base URL to point at the background service.
const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8080').replace(/\/$/, '');
const BACKEND_OFFLINE_MESSAGE = `Background API offline at ${API_BASE}. Camera preview still works; inference needs the background service.`;

// Dashboard data shapes -----------------------------------------------------
// These defaults mirror the backend JSON so components can render safely while
// the first /status request is still loading or when the API is offline.
const emptyTiming = {
  fps: 0,
  preprocess_ms: 0,
  hsv_ms: 0,
  inference_ms: 0,
  total_ms: 0,
};

const emptyModelFeed = {
  loaded: false,
  has_result: false,
  message: 'Model status unavailable',
  timing: emptyTiming,
  detections: [],
  major_defect_detected: false,
  frame_index: 0,
  updated_at: '',
};

// Frontend-only state lives beside the latest backend status in one object.
const initialDashboard = {
  apiOnline: false,
  backendMessage: BACKEND_OFFLINE_MESSAGE,
  browserCameraRunning: false,
  browserCameraStream: null,
  cameras: [{ id: '', label: 'Default camera' }],
  error: '',
  frameToken: Date.now(),
  modelFrames: {
    rgb: true,
    hsv: true,
  },
  selectedCameraId: '',
  status: null,
};

// Model feed configuration --------------------------------------------------
// Keep feed labels, image endpoints, and empty-state copy in one place so the
// RGB/HSV UI stays symmetrical.
const modelFeeds = [
  {
    id: 'rgb',
    label: 'RGB',
    endpoint: '/rgb-frame.jpg',
    missing: 'Load the RGB ONNX model before this feed can run.',
  },
  {
    id: 'hsv',
    label: 'HSV',
    endpoint: '/hsv-frame.jpg',
    missing: 'Load the HSV ONNX model before this feed can run.',
  },
];

// Small display helpers -----------------------------------------------------
const stats = [
  { label: 'FPS', read: (feed) => fmt(feed.timing?.fps) },
  { label: 'Total ms', read: (feed) => fmt(feed.timing?.total_ms) },
  { label: 'Infer ms', read: (feed) => fmt(feed.timing?.inference_ms) },
  { label: 'Detections', read: (feed) => feed.detections?.length ?? 0 },
];

function fmt(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '0.0';
}

function toErrorMessage(exc) {
  return exc instanceof Error ? exc.message : String(exc);
}

function withToken(path, token) {
  return `${API_BASE}${path}?t=${token}`;
}

function latestModelFrameIndex(status) {
  if (!status?.models) {
    return 0;
  }
  return modelFeeds.reduce((latest, { id }) => {
    const frameIndex = Number(status.models?.[id]?.frame_index || 0);
    return Math.max(latest, Number.isFinite(frameIndex) ? frameIndex : 0);
  }, 0);
}

function nextModelFrameFlags(currentFrames, statusBody) {
  let changed = false;
  const nextFrames = { ...currentFrames };
  for (const { id } of modelFeeds) {
    const hasFrame = Boolean(statusBody.inference_running && statusBody.models?.[id]?.has_result);
    if (hasFrame && !nextFrames[id]) {
      nextFrames[id] = true;
      changed = true;
    }
  }
  return changed ? nextFrames : currentFrames;
}

function normalizeCameras(devices) {
  const videoDevices = devices.filter((device) => device.kind === 'videoinput');
  if (videoDevices.length === 0) {
    return [{ id: '', label: 'Default camera' }];
  }
  return videoDevices.map((device, index) => ({
    id: device.deviceId,
    label: device.label || `Camera ${index}`,
  }));
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function postJson(path, body) {
  return requestJson(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function modelState(status, id) {
  return status?.models?.[id] || emptyModelFeed;
}

function stopStream(stream) {
  stream?.getTracks().forEach((track) => track.stop());
}

export default function App() {
  // State and derived dashboard values -------------------------------------
  const [dashboard, setDashboard] = useState(initialDashboard);
  const refreshInFlight = useRef(false);
  const streamRef = useRef(null);

  const {
    apiOnline,
    backendMessage,
    browserCameraRunning,
    browserCameraStream,
    cameras,
    error,
    frameToken,
    modelFrames,
    selectedCameraId,
    status,
  } = dashboard;

  const backendCameraRunning = Boolean(status?.camera_running);
  const inferenceRunning = apiOnline && Boolean(status?.inference_running);
  const cameraRunning = browserCameraRunning;
  const activeModel = status?.active_model || 'rgb';
  const selectedCamera = cameras.find((camera) => camera.id === selectedCameraId) || cameras[0];
  const hasMajorDefect = inferenceRunning && modelFeeds.some(({ id }) => modelState(status, id).major_defect_detected);

  // Browser camera discovery ------------------------------------------------
  const loadCameras = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setDashboard((current) => ({
        ...current,
        cameras: [{ id: '', label: 'Default camera' }],
        selectedCameraId: '',
      }));
      return;
    }

    try {
      const nextCameras = normalizeCameras(await navigator.mediaDevices.enumerateDevices());
      setDashboard((current) => {
        const currentSelectionExists = nextCameras.some((camera) => camera.id === current.selectedCameraId);
        return {
          ...current,
          cameras: nextCameras,
          selectedCameraId: currentSelectionExists ? current.selectedCameraId : nextCameras[0]?.id || '',
        };
      });
    } catch (exc) {
      setDashboard((current) => ({ ...current, error: toErrorMessage(exc) }));
    }
  }, []);

  // Backend polling ---------------------------------------------------------
  // Polling updates JSON every second, but image URLs are cache-busted only
  // when a model feed reports a new frame. That avoids re-downloading the same
  // JPEGs while the backend is paused or only the camera preview is active.
  const refresh = useCallback(async () => {
    if (refreshInFlight.current) return;
    refreshInFlight.current = true;
    try {
      const statusBody = await requestJson('/status');
      setDashboard((current) => {
        const currentFrame = latestModelFrameIndex(current.status);
        const nextFrame = latestModelFrameIndex(statusBody);
        const nextFrames = nextModelFrameFlags(current.modelFrames, statusBody);
        return {
          ...current,
          apiOnline: true,
          backendMessage: '',
          status: statusBody,
          modelFrames: nextFrames,
          frameToken: nextFrame !== currentFrame || nextFrames !== current.modelFrames ? Date.now() : current.frameToken,
        };
      });
    } catch {
      setDashboard((current) => ({
        ...current,
        apiOnline: false,
        backendMessage: BACKEND_OFFLINE_MESSAGE,
        status: null,
      }));
    } finally {
      refreshInFlight.current = false;
    }
  }, []);

  useEffect(() => {
    refresh();
    loadCameras();
    const id = window.setInterval(refresh, 1000);
    return () => {
      window.clearInterval(id);
      stopStream(streamRef.current);
    };
  }, [loadCameras, refresh]);

  // User actions ------------------------------------------------------------
  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setDashboard((current) => ({
        ...current,
        error: 'This browser does not expose camera access. Try Chrome, Edge, or another modern browser on localhost.',
      }));
      return;
    }

    try {
      const video = selectedCameraId ? { deviceId: { exact: selectedCameraId } } : true;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video });
      stopStream(streamRef.current);
      streamRef.current = stream;
      const deviceId = stream.getVideoTracks()[0]?.getSettings().deviceId || selectedCameraId;

      setDashboard((current) => ({
        ...current,
        browserCameraRunning: true,
        browserCameraStream: stream,
        error: '',
        selectedCameraId: deviceId || current.selectedCameraId,
      }));

      await loadCameras();
    } catch (exc) {
      setDashboard((current) => ({
        ...current,
        error: `Camera could not start: ${toErrorMessage(exc)}`,
      }));
    }
  }, [loadCameras, selectedCameraId]);

  const stopCamera = useCallback(async () => {
    stopStream(streamRef.current);
    streamRef.current = null;
    setDashboard((current) => ({
      ...current,
      browserCameraRunning: false,
      browserCameraStream: null,
    }));

    if (apiOnline) {
      try {
        await requestJson('/stop-camera', { method: 'POST' });
        await refresh();
      } catch {
        setDashboard((current) => ({
          ...current,
          apiOnline: false,
          backendMessage: BACKEND_OFFLINE_MESSAGE,
          status: null,
        }));
      }
    }
  }, [apiOnline, refresh]);

  const startInference = useCallback(async () => {
    if (!browserCameraRunning) {
      setDashboard((current) => ({ ...current, error: 'Start the camera before running inference.' }));
      return;
    }
    if (!apiOnline) {
      setDashboard((current) => ({
        ...current,
        error: `Inference needs the C++ background service running at ${API_BASE}.`,
      }));
      return;
    }

    try {
      const cameraIndex = Math.max(0, cameras.findIndex((camera) => camera.id === selectedCameraId));
      await postJson('/set-camera', { camera: cameraIndex });
      await requestJson('/start-camera', { method: 'POST' });
      await requestJson('/start-inference', { method: 'POST' });
      await refresh();
    } catch (exc) {
      setDashboard((current) => ({ ...current, error: toErrorMessage(exc) }));
    }
  }, [apiOnline, browserCameraRunning, cameras, refresh, selectedCameraId]);

  const stopInference = useCallback(async () => {
    if (!apiOnline) return;
    try {
      await requestJson('/stop-inference', { method: 'POST' });
      await refresh();
    } catch (exc) {
      setDashboard((current) => ({ ...current, error: toErrorMessage(exc) }));
    }
  }, [apiOnline, refresh]);

  const setActiveModel = useCallback(async (model) => {
    if (!apiOnline || model === activeModel) return;
    try {
      await postJson('/set-model', { model });
      setDashboard((current) => (
        current.status
          ? { ...current, error: '', status: { ...current.status, active_model: model } }
          : { ...current, error: '' }
      ));
      await refresh();
    } catch (exc) {
      setDashboard((current) => ({ ...current, error: toErrorMessage(exc) }));
    }
  }, [activeModel, apiOnline, refresh]);

  const setCamera = useCallback((event) => {
    setDashboard((current) => ({
      ...current,
      error: '',
      selectedCameraId: event.currentTarget.value,
    }));
  }, []);

  const refreshAll = useCallback(async () => {
    await Promise.all([refresh(), loadCameras()]);
  }, [loadCameras, refresh]);

  const setModelFrameAvailable = useCallback((id, available) => {
    setDashboard((current) => (
      current.modelFrames[id] === available
        ? current
        : { ...current, modelFrames: { ...current.modelFrames, [id]: available } }
    ));
  }, []);

  // Summary strip -----------------------------------------------------------
  const summary = useMemo(
    () => [
      { label: 'Camera', value: cameraRunning ? selectedCamera?.label || 'Live' : 'Stopped' },
      { label: 'Inference', value: inferenceRunning ? 'Running' : apiOnline ? 'Stopped' : 'API offline' },
      { label: 'RGB', value: apiOnline ? (modelState(status, 'rgb').loaded ? 'Loaded' : 'Missing') : 'API offline' },
      { label: 'HSV', value: apiOnline ? (modelState(status, 'hsv').loaded ? 'Loaded' : 'Missing') : 'API offline' },
    ],
    [apiOnline, cameraRunning, inferenceRunning, selectedCamera?.label, status],
  );

  return (
    <main className="app-shell">
      <DashboardHeader
        cameraRunning={cameraRunning}
        inferenceRunning={inferenceRunning}
      />
      <Controls
        activeModel={activeModel}
        apiOnline={apiOnline}
        cameraRunning={cameraRunning}
        cameras={cameras}
        inferenceRunning={inferenceRunning}
        selectedCameraId={selectedCameraId}
        onRefresh={refreshAll}
        onSetCamera={setCamera}
        onSetModel={setActiveModel}
        onStartCamera={startCamera}
        onStartInference={startInference}
        onStopCamera={stopCamera}
        onStopInference={stopInference}
      />

      {backendMessage && <section className="notice-banner" role="status">{backendMessage}</section>}

      <section className="summary-strip" aria-label="Inspection summary">
        {summary.map((item) => (
          <SummaryItem key={item.label} label={item.label} value={item.value} />
        ))}
      </section>

      {hasMajorDefect && (
        <section className="alert-banner" role="alert">
          <AlertTriangle size={20} aria-hidden="true" />
          <strong>Major defect detected</strong>
        </section>
      )}

      {error && <section className="error-banner" role="status">{error}</section>}

      <CameraFeed
        cameraRunning={cameraRunning}
        stream={browserCameraStream}
      />

      <section className="model-grid" aria-label="Model feeds">
        {modelFeeds.map((feed) => (
          <ModelFeed
            apiOnline={apiOnline}
            available={modelFrames[feed.id]}
            backendCameraRunning={backendCameraRunning}
            config={feed}
            cameraRunning={cameraRunning}
            frameToken={frameToken}
            inferenceRunning={inferenceRunning}
            key={feed.id}
            state={modelState(status, feed.id)}
            onFrameState={setModelFrameAvailable}
          />
        ))}
      </section>
    </main>
  );
}

// Presentational components -------------------------------------------------
const DashboardHeader = memo(function DashboardHeader({ cameraRunning, inferenceRunning }) {
  return (
    <header className="topbar">
      <hgroup>
        <p className="eyebrow">Naartjie defect system</p>
        <h1>Camera and Model Inspection</h1>
      </hgroup>
      <output className={`status-pill ${cameraRunning ? 'is-running' : ''}`} aria-live="polite">
        {cameraRunning ? (inferenceRunning ? 'Camera + inference' : 'Camera only') : 'Stopped'}
      </output>
    </header>
  );
});

const Controls = memo(function Controls({
  activeModel,
  apiOnline,
  cameraRunning,
  cameras,
  inferenceRunning,
  selectedCameraId,
  onRefresh,
  onSetCamera,
  onSetModel,
  onStartCamera,
  onStartInference,
  onStopCamera,
  onStopInference,
}) {
  return (
    <section className="control-row" aria-label="Inspection controls">
      <label className="camera-picker">
        <span>Camera</span>
        <select disabled={cameraRunning} value={selectedCameraId} onChange={onSetCamera}>
          {cameras.map(({ id, label }) => (
            <option key={id || 'default'} value={id}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <fieldset className="model-toggle">
        <legend>Model</legend>
        <div className="model-toggle-options">
          {modelFeeds.map(({ id, label }) => (
            <button
              className={activeModel === id ? 'is-active' : ''}
              disabled={!apiOnline}
              key={id}
              type="button"
              aria-pressed={activeModel === id}
              onClick={() => onSetModel(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </fieldset>
      <button className="command start" type="button" onClick={onStartCamera} disabled={cameraRunning}>
        <Play size={18} aria-hidden="true" />
        <span>Start Camera</span>
      </button>
      <button className="command stop" type="button" onClick={onStopCamera} disabled={!cameraRunning}>
        <CircleStop size={18} aria-hidden="true" />
        <span>Stop Camera</span>
      </button>
      <button
        className="command start"
        type="button"
        onClick={onStartInference}
        disabled={!cameraRunning || inferenceRunning}
      >
        <Play size={18} aria-hidden="true" />
        <span>Start Inference</span>
      </button>
      <button className="command stop" type="button" onClick={onStopInference} disabled={!inferenceRunning}>
        <CircleStop size={18} aria-hidden="true" />
        <span>Stop Inference</span>
      </button>
      <button className="icon-command" type="button" onClick={onRefresh} aria-label="Refresh">
        <RefreshCw size={18} aria-hidden="true" />
      </button>
    </section>
  );
});

const SummaryItem = memo(function SummaryItem({ label, value }) {
  return (
    <p className="summary-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </p>
  );
});

const CameraFeed = memo(function CameraFeed({ cameraRunning, stream }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <section className="camera-section" aria-label="Camera footage">
      <header className="feed-heading">
        <h2>Camera Footage</h2>
        <span>{cameraRunning ? 'Live' : 'Stopped'}</span>
      </header>
      <figure className="camera-frame">
        {cameraRunning && stream ? (
          <video ref={videoRef} aria-label="Live naartjie camera footage" autoPlay muted playsInline />
        ) : (
          <figcaption className="feed-empty">Camera stopped</figcaption>
        )}
      </figure>
    </section>
  );
});

const ModelFeed = memo(function ModelFeed({
  apiOnline,
  available,
  backendCameraRunning,
  cameraRunning,
  config,
  frameToken,
  inferenceRunning,
  onFrameState,
  state,
}) {
  const frameReady = apiOnline && inferenceRunning && state.loaded && state.has_result && available;
  const frameSrc = withToken(config.endpoint, frameToken);
  const statusText = !apiOnline
    ? 'API offline'
    : !state.loaded
      ? 'Model missing'
      : !inferenceRunning
        ? 'Stopped'
        : state.has_result
          ? 'Running'
          : 'Waiting';
  const placeholder = !apiOnline
    ? 'Start the background service to run inference'
    : !state.loaded
      ? config.missing
      : !cameraRunning
        ? 'Start the camera before running inference'
        : !backendCameraRunning
          ? 'Start inference to connect the background camera'
          : !inferenceRunning
            ? 'Start inference to run this model'
            : 'Waiting for model output';

  return (
    <article className="model-feed">
      <header className="feed-heading">
        <h2>{config.label} Model</h2>
        <span className={state.loaded && apiOnline ? 'feed-badge loaded' : 'feed-badge missing'}>{statusText}</span>
      </header>

      <figure className="model-frame">
        {frameReady ? (
          <img
            src={frameSrc}
            alt={`${config.label} model naartjie defect footage`}
            onError={() => onFrameState(config.id, false)}
            onLoad={() => onFrameState(config.id, true)}
          />
        ) : (
          <figcaption className={state.loaded && apiOnline ? 'feed-empty' : 'feed-warning'}>
            {placeholder}
            {apiOnline && state.message && <small>{state.message}</small>}
          </figcaption>
        )}
      </figure>

      <dl className="feed-stats">
        {stats.map(({ label, read }) => (
          <div className="stat" key={label}>
            <dt>{label}</dt>
            <dd>{read(state)}</dd>
          </div>
        ))}
      </dl>
    </article>
  );
});
