import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
} from 'react';
import {
  Activity,
  AlertCircle,
  BrainCircuit,
  Camera,
  CameraOff,
  CheckCircle2,
  ImagePlus,
  Languages,
  ShieldCheck,
  Trash2,
  Undo2,
  Upload,
  Volume2,
} from 'lucide-react';

const API = import.meta.env.VITE_API_BASE_URL || '';

type Alternative = {
  label: string;
  confidence: number;
};

type Prediction = {
  status?: string;
  label?: string;
  confidence?: number;
  alternatives?: Alternative[];
  message?: string;
};

type Health = {
  status?: string;
  model_ready?: boolean;
  language?: string;
  runtime?: string;
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [prediction, setPrediction] = useState<Prediction>({
    status: 'waiting',
  });
  const [history, setHistory] = useState<string[]>([]);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');

  // Fetch backend/model health once when the app mounts.
  useEffect(() => {
    let cancelled = false;

    fetch(`${API}/api/v1/health`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Health check failed (${response.status}).`);
        }
        return (await response.json()) as Health;
      })
      .then((result) => {
        if (!cancelled) setHealth(result);
      })
      .catch(() => {
        if (!cancelled) {
          setHealth({ status: 'offline', model_ready: false });
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // Revoke the previous uploaded-image URL when it is replaced or on unmount.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const cameraStartingRef = useRef(false);

  const stopCamera = useCallback(() => {
    const stream = streamRef.current;
    streamRef.current = null;

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    // Detaching the stream is sufficient; do not call video.pause().
    // Calling pause() while play() is pending can produce an AbortError.
    const video = videoRef.current;
    if (video) {
      video.srcObject = null;
    }

    setCameraOn(false);
    setCameraReady(false);
  }, []);

  // Stop the webcam if the component is unmounted.
  useEffect(() => {
    return () => {
      const stream = streamRef.current;
      streamRef.current = null;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function startCamera() {
    if (cameraStartingRef.current || streamRef.current) return;

    cameraStartingRef.current = true;
    setError('');
    setCameraReady(false);

    const video = videoRef.current;
    let stream: MediaStream | null = null;

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          'Camera access is unavailable. Open the app on localhost or HTTPS and use a supported browser.',
        );
      }

      if (!video) {
        throw new Error('Camera preview element was not found.');
      }

      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      // If the component was unmounted or another stream took ownership,
      // do not attach this stream.
      streamRef.current = stream;

      const track = stream.getVideoTracks()[0];
      if (!track || track.readyState !== 'live') {
        throw new Error('The browser did not provide an active camera track.');
      }

      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;
      video.srcObject = stream;
      setCameraOn(true);

      // Wait for metadata before calling play(). Register listeners first
      // and check readyState again to avoid missing a fast event.
      if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
        await new Promise<void>((resolve, reject) => {
          const timeoutId = window.setTimeout(() => {
            cleanup();
            reject(new Error('Timed out while waiting for camera video metadata.'));
          }, 10000);

          const cleanup = () => {
            window.clearTimeout(timeoutId);
            video.removeEventListener('loadedmetadata', onLoadedMetadata);
            video.removeEventListener('error', onVideoError);
          };

          const onLoadedMetadata = () => {
            cleanup();
            resolve();
          };

          const onVideoError = () => {
            cleanup();
            reject(new Error('The browser could not read the camera video stream.'));
          };

          video.addEventListener('loadedmetadata', onLoadedMetadata);
          video.addEventListener('error', onVideoError);

          if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
            cleanup();
            resolve();
          }
        });
      }

      // Confirm this is still the active stream before playback.
      if (streamRef.current !== stream || track.readyState !== 'live') {
        throw new Error('The camera stream stopped before playback began.');
      }

      // Do not call pause() or replace srcObject while this promise is pending.
      await video.play();

      // Wait until an actual frame is available.
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        await new Promise<void>((resolve, reject) => {
          const timeoutId = window.setTimeout(() => {
            cleanup();
            reject(new Error('Timed out while waiting for the first camera frame.'));
          }, 10000);

          const cleanup = () => {
            window.clearTimeout(timeoutId);
            video.removeEventListener('loadeddata', onLoadedData);
            video.removeEventListener('error', onVideoError);
          };

          const onLoadedData = () => {
            cleanup();
            resolve();
          };

          const onVideoError = () => {
            cleanup();
            reject(new Error('The camera frame could not be rendered.'));
          };

          video.addEventListener('loadeddata', onLoadedData);
          video.addEventListener('error', onVideoError);

          if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            cleanup();
            resolve();
          }
        });
      }

      if (streamRef.current !== stream) return;

      if (video.videoWidth === 0 || video.videoHeight === 0) {
        throw new Error('The camera connected, but no video frames were received.');
      }

      setCameraReady(true);
      console.info('SignBridge camera is ready:', {
        width: video.videoWidth,
        height: video.videoHeight,
        track: track.getSettings(),
      });
    } catch (err) {
      // A stream may have been intentionally stopped while startup was pending.
      // In that case, do not show an interruption error to the user.
      const isCurrentAttempt = streamRef.current === stream;

      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }

      if (isCurrentAttempt) {
        streamRef.current = null;
        if (video && video.srcObject === stream) {
          video.srcObject = null;
        }
        setCameraOn(false);
        setCameraReady(false);

        const errorName = err instanceof DOMException ? err.name : '';
        if (errorName !== 'AbortError') {
          setError(
            err instanceof Error
              ? err.message
              : 'Camera access was not available.',
          );
        }
      }

      console.error('SignBridge camera initialization failed:', err);
    } finally {
      cameraStartingRef.current = false;
    }
  }

  async function sendImage(file: File) {
    setBusy(true);
    setError('');

    try {
      const form = new FormData();
      form.append('file', file);

      const response = await fetch(`${API}/api/v1/predict/image`, {
        method: 'POST',
        body: form,
      });

      const result = (await response.json()) as Prediction & {
        detail?: string;
      };

      if (!response.ok) {
        throw new Error(
          result.message ||
            result.detail ||
            `Image prediction failed (${response.status}).`,
        );
      }

      setPrediction(result);

      if (result.status === 'ok' && result.label) {
        setHistory((items) => [result.label!, ...items].slice(0, 20));
        setMessage((current) =>
          current ? `${current}${result.label}` : result.label!,
        );
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not process this image.',
      );
    } finally {
      setBusy(false);
    }
  }

  async function captureSign() {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (
      !cameraOn ||
      !cameraReady ||
      !video ||
      !canvas ||
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      setError('Start the camera and wait until the live preview is ready before capturing.');
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');
    if (!context) {
      setError('Could not prepare the canvas for camera capture.');
      return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, 'image/jpeg', 0.92);
    });

    if (!blob) {
      setError('Could not capture the camera frame.');
      return;
    }

    await sendImage(
      new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' }),
    );
  }

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setPreviewUrl(URL.createObjectURL(file));
    await sendImage(file);

    // Allows selecting the same file again later.
    event.target.value = '';
  }

  function speakMessage() {
    if (!message.trim()) return;

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(
      new SpeechSynthesisUtterance(message),
    );
  }

  function clearMessage() {
    setHistory([]);
    setMessage('');
    setPrediction({ status: 'waiting' });
  }

  function undoLetter() {
    setHistory((items) => items.slice(1));
    setMessage((value) => value.slice(0, -1));
  }

  return (
    <main className="shell">
      <nav className="nav">
        <div className="brand">
          <span className="logo">
            <BrainCircuit size={21} />
          </span>
          <span>
            SignBridge <b>AI</b>
            <small>ACCESSIBILITY THROUGH AI</small>
          </span>
        </div>

        <div className="navstatus">
          <span className={`dot ${health?.model_ready ? 'good' : ''}`} />
          {health?.model_ready ? 'MODEL READY' : 'MODEL NOT LOADED'}
          <span className="divider" />
          {health?.language || 'ISL'}
        </div>
      </nav>

      <section className="hero">
        <div>
          <div className="eyebrow">
            <Activity size={15} /> STATIC SIGN IMAGE RECOGNITION
          </div>
          <h1>
            Make every gesture
            <br />
            <em>understood.</em>
          </h1>
          <p className="sub">
            Recognize static Indian Sign Language alphabet signs from a
            webcam snapshot or an uploaded image. No video dataset or
            continuous video stream is required.
          </p>
          <div className="pills">
            <span>
              <ShieldCheck size={14} /> Images processed by your API
            </span>
            <span>
              <Languages size={14} /> ISL alphabet A–Z
            </span>
          </div>
        </div>

        <div className="hero-card">
          <div className="card-top">
            <span>MODEL STATUS</span>
            <span className="live">
              <i /> SYSTEM CHECK
            </span>
          </div>
          <div className="model-state">
            <div className="model-icon">
              <BrainCircuit />
            </div>
            <div>
              <b>
                {health?.model_ready
                  ? 'Ready for recognition'
                  : 'Awaiting trained checkpoint'}
              </b>
              <small>
                {health?.runtime || 'ONNX Runtime / CPU'} ·{' '}
                {health?.language || 'ISL'}
              </small>
            </div>
          </div>
          <div className="meter">
            <span
              style={{
                width: health?.model_ready ? '100%' : '8%',
              }}
            />
          </div>
          <small className="muted">
            {health?.model_ready
              ? 'Image model loaded successfully.'
              : 'Download the image dataset, train MobileNetV3-Small and export the ONNX model.'}
          </small>
        </div>
      </section>

      <section className="workspace">
        <div className="section-head">
          <div>
            <span className="eyebrow">WORKSPACE / 01</span>
            <h2>Recognize a sign</h2>
          </div>
          <span className="tag">
            <span className={`dot ${cameraOn ? 'good' : ''}`} />
            {cameraOn
              ? cameraReady
                ? 'CAMERA ACTIVE'
                : 'CAMERA STARTING'
              : 'CAMERA STANDBY'}
          </span>
        </div>

        <div className="grid">
          <div className="image-card">
            <div className="video-head">
              <span>
                <Camera size={16} /> CAMERA / IMAGE INPUT
              </span>
              <span className="muted">JPG · PNG · WEBP</span>
            </div>

            <div
              className="preview-wrap"
              style={{
                position: 'relative',
                width: '100%',
                height: 360,
                minHeight: 360,
                overflow: 'hidden',
                background: '#111827',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {/* Keep this video element mounted; only toggle its visibility. */}
              <video
                ref={videoRef}
                muted
                playsInline
                autoPlay
                className="camera-preview"
                style={{
                  display: cameraOn ? 'block' : 'none',
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                  background: '#111827',
                }}
              />

              {!cameraOn && previewUrl && (
                <img
                  src={previewUrl}
                  alt="Selected sign"
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                  }}
                />
              )}

              {!cameraOn && !previewUrl && (
                <div className="placeholder">
                  <div className="camera-icon">
                    <ImagePlus size={25} />
                  </div>
                  <b>Your sign image appears here</b>
                  <span>
                    Start the camera or upload a clear hand-sign image
                  </span>
                </div>
              )}

              {cameraOn && (
                <span className="camera-label">
                  {cameraReady ? 'LIVE CAMERA PREVIEW' : 'STARTING CAMERA…'}
                </span>
              )}
            </div>

            <canvas ref={canvasRef} className="hidden" />

            <div className="actions">
              {cameraOn ? (
                <button className="secondary" onClick={stopCamera}>
                  <CameraOff size={16} /> Stop camera
                </button>
              ) : (
                <button className="secondary" onClick={startCamera}>
                  <Camera size={16} /> Start camera
                </button>
              )}

              <button
                className="primary"
                onClick={captureSign}
                disabled={!cameraReady || busy}
              >
                <CheckCircle2 size={16} /> Capture &amp; recognize
              </button>

              <button
                className="secondary"
                onClick={() => fileRef.current?.click()}
                disabled={busy}
              >
                <Upload size={16} /> Upload image
              </button>

              <input
                ref={fileRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFile}
                hidden
              />
            </div>

            {busy && (
              <div className="progress-note">
                Running image classification…
              </div>
            )}

            {error && (
              <div className="error">
                <AlertCircle size={16} /> {error}
              </div>
            )}
          </div>

          <div className="result-column">
            <div className="result-card">
              <div className="card-top">
                <span>RECOGNITION OUTPUT</span>
                <span className="muted">TOP 3</span>
              </div>

              <div className="prediction">
                <small>PREDICTED LETTER</small>
                <h3>
                  {prediction.label ||
                    (prediction.status === 'unknown' ? 'Unknown' : '—')}
                </h3>
                <span className="confidence">
                  {typeof prediction.confidence === 'number'
                    ? `${(prediction.confidence * 100).toFixed(1)}% confidence`
                    : prediction.message ||
                      'Capture or upload an image to begin'}
                </span>
              </div>

              <div className="alternatives">
                <b>ALTERNATIVE PREDICTIONS</b>
                {prediction.alternatives?.map((item, index) => (
                  <div className="alt" key={item.label}>
                    <span>
                      {index + 1}. {item.label}
                    </span>
                    <span>
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
                {!prediction.alternatives?.length && (
                  <p className="muted">
                    Top alternatives appear after inference.
                  </p>
                )}
              </div>
            </div>

            <div className="history-card">
              <div className="card-top">
                <span>RECENT LETTERS</span>
                <button
                  className="icon-btn"
                  onClick={clearMessage}
                  title="Clear history"
                >
                  <Trash2 size={15} />
                </button>
              </div>

              {history.length ? (
                history.slice(0, 8).map((letter, index) => (
                  <div
                    className="history-item"
                    key={`${letter}-${index}`}
                  >
                    <span className="history-num">
                      {String(index + 1).padStart(2, '0')}
                    </span>
                    {letter}
                  </div>
                ))
              ) : (
                <p className="muted">
                  Recognized letters will appear here.
                </p>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="translation">
        <div className="section-head">
          <div>
            <span className="eyebrow">WORKSPACE / 02</span>
            <h2>Build your word</h2>
          </div>
          <span className="tag">EDITABLE TEXT</span>
        </div>

        <textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          placeholder="Recognized letters are appended here. Edit the text as needed…"
        />

        <div className="translation-actions">
          <small className="muted">
            This spells words from static alphabet signs; it is not
            automatic ISL sentence translation.
          </small>
          <div className="actions compact">
            <button
              className="secondary"
              onClick={undoLetter}
              disabled={!message}
            >
              <Undo2 size={15} /> Undo
            </button>
            <button
              className="primary"
              onClick={speakMessage}
              disabled={!message.trim()}
            >
              <Volume2 size={16} /> Speak
            </button>
          </div>
        </div>
      </section>

      <footer>
        <span>
          SignBridge AI <b>·</b> Image-only research starter
        </span>
        <span>MobileNetV3-Small · ONNX Runtime · ISL alphabet</span>
      </footer>
    </main>
  );
}
