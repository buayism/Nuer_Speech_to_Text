import html
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time

import imageio_ffmpeg
import librosa
import streamlit as st
import streamlit.components.v1 as components
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoProcessor, Wav2Vec2ForCTC

st.set_page_config(
    page_title="Nuer MMS Speech-to-Text", page_icon="🎙️", layout="centered"
)

BASE_MODEL_ID = "facebook/mms-1b-all"
ADAPTER_REPO = "buayism/nuer_mms_adapter"
ADAPTER_FILE = "adapter.nus.safetensors"
SAMPLE_RATE = 16000
TOTAL_STEPS = 6
BAR_DELAYS = [0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0]

# ---------- Styling (Dark studio theme + custom buttons) ----------
st.markdown(
    """
<style>
.block-container { padding-top: 1.5rem; max-width: 780px; }

@keyframes fadeUp { from {opacity:0; transform: translateY(14px);} to {opacity:1; transform:none;} }
@keyframes pulse {
  0%,100% { transform: scale(1); filter: drop-shadow(0 0 0 rgba(255,75,75,0)); }
  50% { transform: scale(1.1); filter: drop-shadow(0 0 14px rgba(255,75,75,.75)); }
}
@keyframes bounce { 0%,100% { height: 8px; } 50% { height: 44px; } }

.hero { text-align:center; animation: fadeUp .6s ease both; }
.hero .mic { font-size: 3.2rem; display:inline-block; animation: pulse 2.4s ease-in-out infinite; }
.hero .title {
  font-size: 2.4rem; font-weight: 800; line-height: 1.15; margin: .2rem 0 .4rem;
  background: linear-gradient(90deg, #ff4b4b, #ffffff);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero .sub { opacity: .8; margin-bottom: .8rem; font-size: 0.95rem; }

.steps { display:flex; gap:.6rem; justify-content:center; flex-wrap:wrap; margin: .5rem 0 1.2rem; animation: fadeUp .8s ease both; }
.step { padding:.35rem .85rem; border-radius:999px; background:#262730; font-size:.82rem; border:1px solid rgba(255,75,75,.35); }
.step b { color:#ff4b4b; margin-right:.3rem; }

[data-testid="stFileUploader"] section {
  border: 2px dashed rgba(255,75,75,.55); border-radius: 16px; background: #262730;
  padding: 1.2rem; transition: all .25s ease;
}
[data-testid="stFileUploader"] section:hover {
  border-color: #ff4b4b; box-shadow: 0 0 24px rgba(255,75,75,.25); transform: translateY(-2px);
}

div.stButton > button, button[data-testid="stBaseButton-primary"] {
  width: 100%; padding: .75rem 1rem; font-size: 1.1rem; font-weight: 700;
  border-radius: 12px; transition: all .2s ease;
  box-shadow: 0 0 18px rgba(255,75,75,.35);
}
div.stButton > button:hover, button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); box-shadow: 0 0 30px rgba(255,75,75,.6);
}

.loader {
  text-align:center; padding:1.4rem; background:#262730; border-radius:16px;
  border:1px solid rgba(255,75,75,.4); box-shadow: 0 0 24px rgba(255,75,75,.15);
}
.bars { display:flex; gap:5px; justify-content:center; align-items:center; height:50px; }
.bars span { width:6px; height:8px; background:#ff4b4b; border-radius:3px; animation: bounce 1s ease-in-out infinite; }
.loader .step-tag { margin-top:.7rem; font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#ff4b4b; }
.loader .msg { margin-top:.3rem; font-size:1.15rem; font-weight:700; }
.loader .note { margin-top:.3rem; opacity:.8; font-size:.92rem; }
.pbar { height:8px; border-radius:999px; background:rgba(255,255,255,.1); margin-top:.9rem; overflow:hidden; }
.pfill { height:100%; background:#ff4b4b; border-radius:999px; box-shadow:0 0 12px rgba(255,75,75,.7); transition: width .25s ease; }

.result {
  background: rgba(33,195,84,.1); border: 1px solid rgba(61,213,109,.5);
  border-radius: 16px; padding: 1.2rem 1.4rem; margin-top: 1rem;
  box-shadow: 0 0 24px rgba(33,195,84,.15); animation: fadeUp .5s ease both;
}
.result .label { color:#3dd56d; font-size:.8rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.result .text {
  font-family: "Times New Roman", Times, "Liberation Serif", serif;
  font-size: 1.8rem; font-weight: 700; line-height: 1.5; margin-top:.4rem;
}

[data-testid="stCode"] pre, [data-testid="stCode"] code {
  font-family: "Times New Roman", Times, "Liberation Serif", serif !important;
  font-size: 1.15rem !important;
  white-space: pre-wrap !important;
}

.hint { text-align:center; opacity:.65; padding: .8rem 0; }
.footer { text-align:center; opacity:.5; font-size:.8rem; margin-top:2.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
<div class="hero">
<div class="mic">🎙️</div>
<div class="title">Nuer Speech-to-Text</div>
<div class="sub">Turn spoken Nuer into text, powered by your fine-tuned MMS adapter.</div>
</div>
<div class="steps">
<div class="step"><b>1</b> Record or Upload</div>
<div class="step"><b>2</b> Click Transcribe</div>
<div class="step"><b>3</b> Copy or Download</div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Model Loading ----------
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(ADAPTER_REPO)

    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL_ID,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    adapter_path = hf_hub_download(ADAPTER_REPO, ADAPTER_FILE)
    state_dict = load_file(adapter_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.eval()
    return processor, model


with st.spinner("Warming up the Nuer model..."):
    processor, model = load_model()


# ---------- Audio & Model Processing ----------
def loader_html(step, title, note, elapsed, progress=None):
    phase = elapsed % 1.0
    bars = "".join(
        f'<span style="animation-delay:{d - phase:.2f}s"></span>' for d in BAR_DELAYS
    )
    pbar = ""
    if progress is not None:
        pbar = (
            f'<div class="pbar"><div class="pfill" '
            f'style="width:{progress * 100:.0f}%"></div></div>'
        )
    return (
        f'<div class="loader"><div class="bars">{bars}</div>'
        f'<div class="step-tag">Step {step} of {TOTAL_STEPS} · {elapsed:.0f}s elapsed</div>'
        f'<div class="msg">{html.escape(title)}</div>'
        f'<div class="note">{html.escape(note)}</div>{pbar}</div>'
    )


class Tracker:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.start = time.time()

    def show(self, step, title, note, progress=None):
        self.placeholder.markdown(
            loader_html(step, title, note, time.time() - self.start, progress),
            unsafe_allow_html=True,
        )


def load_audio_16k(audio_bytes, tracker):
    """Convert audio bytes to 16kHz mono WAV using ffmpeg."""
    tracker.show(1, "Reading audio input…", "Preparing recorded audio for processing.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as src:
        src.write(audio_bytes)
        src_path = src.name
    out_path = src_path + "_16k.wav"

    ffmpeg_exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

    try:
        tracker.show(
            2,
            "Converting audio format…",
            "Resampling to 16 kHz mono for model compatibility.",
        )
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", src_path, "-ac", "1", "-ar", str(SAMPLE_RATE), out_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion error:\n{result.stderr[-500:]}")

        speech_array, _ = librosa.load(out_path, sr=SAMPLE_RATE)
    finally:
        for p in (src_path, out_path):
            if os.path.exists(p):
                os.remove(p)

    return speech_array


def attach_progress_hooks(state):
    w2v = model.wav2vec2
    handles = [
        w2v.feature_projection.register_forward_hook(
            lambda m, a, o: state.update(stage="encoder")
        ),
        model.lm_head.register_forward_pre_hook(
            lambda m, a: state.update(stage="head")
        ),
    ]
    for layer in w2v.encoder.layers:
        handles.append(
            layer.register_forward_hook(
                lambda m, a, o: state.update(layer=state["layer"] + 1)
            )
        )
    return handles


def run_model(input_values, result):
    try:
        with torch.inference_mode():
            result["logits"] = model(input_values).logits
    except Exception as e:
        result["error"] = e


def transcribe(speech, tracker):
    chunk = 20 * SAMPLE_RATE if len(speech) > 30 * SAMPLE_RATE else len(speech)
    pieces = [speech[i : i + chunk] for i in range(0, len(speech), chunk)]
    if len(pieces) > 1 and len(pieces[-1]) < SAMPLE_RATE:
        pieces.pop()

    total_layers = len(model.wav2vec2.encoder.layers)
    texts = []

    for idx, piece in enumerate(pieces, 1):
        part = f"Part {idx} of {len(pieces)} · " if len(pieces) > 1 else ""

        tracker.show(
            3,
            "Preparing audio features…",
            part + "Normalizing sound levels.",
        )
        inputs = processor(
            piece, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )

        state = {"stage": "features", "layer": 0}
        handles = attach_progress_hooks(state)
        result = {}
        thread = threading.Thread(
            target=run_model, args=(inputs.input_values, result), daemon=True
        )
        thread.start()
        try:
            while thread.is_alive():
                stage = state["stage"]
                if stage == "features":
                    tracker.show(
                        4,
                        "Extracting acoustic features…",
                        part + "Analyzing pitch and phonetic shapes.",
                    )
                elif stage == "encoder":
                    done = min(state["layer"], total_layers)
                    tracker.show(
                        4,
                        "Analyzing speech frames…",
                        f"{part}Layer {done} of {total_layers}.",
                        progress=done / total_layers,
                    )
                else:
                    tracker.show(
                        5,
                        "Predicting characters…",
                        part + "Evaluating Nuer vocabulary probability matrix.",
                    )
                time.sleep(0.25)
        finally:
            thread.join()
            for h in handles:
                h.remove()

        if "error" in result:
            raise result["error"]

        tracker.show(
            6,
            "Decoding text…",
            part + "Generating final output string.",
        )
        predicted_ids = torch.argmax(result["logits"], dim=-1)
        texts.append(processor.batch_decode(predicted_ids)[0].strip())

    return " ".join(t for t in texts if t)


# ---------- Custom Live Visual Waveform Recorder Component ----------
def render_waveform_recorder():
    """HTML5 canvas live scrolling waveform studio component."""
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          background-color: #1a1c23;
          color: #e0e0e0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 10px;
          border-radius: 16px;
          border: 1px solid rgba(255,75,75,0.25);
        }
        .canvas-container {
          position: relative;
          width: 100%;
          height: 160px;
          background: #121318;
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid #2d2f39;
        }
        canvas {
          width: 100%;
          height: 100%;
          display: block;
        }
        .time-ruler {
          position: absolute;
          top: 4px;
          width: 100%;
          display: flex;
          justify-content: space-between;
          padding: 0 15px;
          font-family: monospace;
          font-size: 11px;
          color: #717585;
          pointer-events: none;
        }
        .timeline-center-line {
          position: absolute;
          top: 50%;
          left: 0;
          width: 100%;
          height: 1px;
          background: rgba(255, 255, 255, 0.08);
          pointer-events: none;
        }
        .playhead-line {
          position: absolute;
          top: 0;
          left: 45%;
          width: 2px;
          height: 100%;
          background: #ff4b4b;
          box-shadow: 0 0 8px rgba(255, 75, 75, 0.8);
          pointer-events: none;
        }
        .controls {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 24px;
          margin-top: 14px;
          width: 100%;
        }
        .btn {
          width: 48px;
          height: 48px;
          border-radius: 50%;
          border: none;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s ease;
          outline: none;
        }
        .btn-rec {
          background: #282a36;
          border: 2px solid #ff4b4b;
        }
        .btn-rec:hover {
          transform: scale(1.08);
          box-shadow: 0 0 15px rgba(255, 75, 75, 0.4);
        }
        .rec-dot {
          width: 16px;
          height: 16px;
          background: #ff4b4b;
          border-radius: 50%;
        }
        .btn-stop {
          background: #282a36;
          border: 2px solid #ffffff;
        }
        .btn-stop:hover {
          transform: scale(1.08);
        }
        .stop-square {
          width: 14px;
          height: 14px;
          background: #ffffff;
          border-radius: 3px;
        }
        .timer {
          font-family: monospace;
          font-size: 1.5rem;
          font-weight: 700;
          color: #ffffff;
          letter-spacing: 1px;
          min-width: 110px;
          text-align: center;
        }
        .recording .rec-dot {
          animation: pulse-dot 1s infinite alternate;
        }
        @keyframes pulse-dot {
          from { transform: scale(0.9); opacity: 0.7; }
          to { transform: scale(1.2); opacity: 1; box-shadow: 0 0 10px #ff4b4b; }
        }
      </style>
    </head>
    <body>
      <div class="canvas-container">
        <div class="time-ruler">
          <span>00:00.00</span>
          <span>00:01.60</span>
          <span>00:03.20</span>
          <span>00:04.80</span>
          <span>00:06.40</span>
        </div>
        <div class="timeline-center-line"></div>
        <canvas id="waveform"></canvas>
        <div class="playhead-line"></div>
      </div>

      <div class="controls">
        <button id="btnStop" class="btn btn-stop" style="opacity: 0.4; cursor: not-allowed;" disabled>
          <div class="stop-square"></div>
        </button>
        <div id="timer" class="timer">00:00.00</div>
        <button id="btnRec" class="btn btn-rec">
          <div class="rec-dot"></div>
        </button>
      </div>

      <script>
        let mediaRecorder;
        let audioChunks = [];
        let audioCtx, analyser, dataArray, source;
        let isRecording = false;
        let startTime;
        let timerInterval;
        let animFrame;

        const canvas = document.getElementById('waveform');
        const ctx = canvas.getContext('2d');
        const timerEl = document.getElementById('timer');
        const btnRec = document.getElementById('btnRec');
        const btnStop = document.getElementById('btnStop');

        let waveHistory = [];
        const PLAYHEAD_X_RATIO = 0.45; // 45% mark from left

        function resizeCanvas() {
          canvas.width = canvas.offsetWidth;
          canvas.height = canvas.offsetHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        function drawWaveform() {
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          const centerY = canvas.height / 2;
          const playheadX = canvas.width * PLAYHEAD_X_RATIO;
          const barWidth = 3;
          const gap = 2;
          const step = barWidth + gap;

          if (isRecording && analyser) {
            analyser.getByteFrequencyData(dataArray);
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
              sum += dataArray[i];
            }
            let avg = sum / dataArray.length;
            let normAmp = Math.min(1.0, Math.max(0.04, (avg / 128.0) * 1.5));
            waveHistory.push(normAmp);
          }

          // Draw scrolling history bars
          for (let i = 0; i < waveHistory.length; i++) {
            let x = playheadX - (waveHistory.length - 1 - i) * step;
            if (x < -10) continue; // Off screen left

            let height = waveHistory[i] * (canvas.height * 0.7);
            ctx.fillStyle = "#8a8d9b";
            ctx.fillRect(x, centerY - height / 2, barWidth, height);
          }

          animFrame = requestAnimationFrame(drawWaveform);
        }

        function updateTimer() {
          const now = Date.now();
          const diff = new Date(now - startTime);
          const mins = String(diff.getUTCMinutes()).padStart(2, '0');
          const secs = String(diff.getUTCSeconds()).padStart(2, '0');
          const ms = String(Math.floor(diff.getUTCMilliseconds() / 10)).padStart(2, '0');
          timerEl.textContent = `${mins}:${secs}.${ms}`;
        }

        btnRec.addEventListener('click', async () => {
          if (isRecording) return;

          try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 128;
            dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            source = audioCtx.createMediaStreamSource(stream);
            source.connect(analyser);

            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            waveHistory = [];

            mediaRecorder.ondataavailable = (e) => {
              if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
              const blob = new Blob(audioChunks, { type: 'audio/wav' });
              const reader = new FileReader();
              reader.readAsDataURL(blob);
              reader.onloadend = () => {
                const base64Audio = reader.result;
                window.parent.postMessage({
                  type: "streamlit:setComponentValue",
                  value: base64Audio
                }, "*");
              };
              stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start(100);
            isRecording = true;
            startTime = Date.now();
            timerInterval = setInterval(updateTimer, 30);

            btnRec.classList.add('recording');
            btnRec.style.opacity = '0.4';
            btnRec.disabled = true;

            btnStop.style.opacity = '1';
            btnStop.disabled = false;
            btnStop.style.cursor = 'pointer';

            drawWaveform();

          } catch (err) {
            alert('Microphone access denied or not supported.');
            console.error(err);
          }
        });

        btnStop.addEventListener('click', () => {
          if (!isRecording) return;
          isRecording = false;
          clearInterval(timerInterval);
          cancelAnimationFrame(animFrame);
          mediaRecorder.stop();

          btnRec.classList.remove('recording');
          btnRec.style.opacity = '1';
          btnRec.disabled = false;

          btnStop.style.opacity = '0.4';
          btnStop.disabled = true;
          btnStop.style.cursor = 'not-allowed';
        });

        // Initialize background canvas view
        drawWaveform();
      </script>
    </body>
    </html>
    """
    return components.html(html_code, height=255)


# ---------- Main UI Layout ----------
tab_record, tab_upload = st.tabs(["🎙️ Record Live Audio", "📁 Upload File"])

audio_bytes = None
file_key = None

with tab_record:
    st.caption("Click the red record button below to speak directly from your browser.")
    recorded_audio_data = render_waveform_recorder()

    # Retrieve base64 wav string from custom html component
    if recorded_audio_data:
        import base64
        try:
            header, encoded = recorded_audio_data.split(",", 1)
            audio_bytes = base64.b64decode(encoded)
            st.audio(audio_bytes, format="audio/wav")
            file_key = f"live_recording_{len(audio_bytes)}"
        except Exception as e:
            st.error(f"Error parsing live recording: {e}")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Drop a recording here or browse",
        type=["wav", "mp3", "m4a", "aac", "ogg"],
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes)
        file_key = f"{uploaded_file.name}-{len(audio_bytes)}"

# Transcribe action trigger
if audio_bytes is not None and file_key is not None:
    if st.button("✨ Transcribe Audio", type="primary"):
        placeholder = st.empty()
        tracker = Tracker(placeholder)
        try:
            speech = load_audio_16k(audio_bytes, tracker)
            if len(speech) == 0:
                raise RuntimeError("The audio recording appears to be empty.")
            text = transcribe(speech, tracker)
        except Exception as e:
            placeholder.empty()
            st.error(f"Something went wrong: {e}")
            st.stop()

        placeholder.empty()
        st.session_state["result"] = {
            "key": file_key,
            "text": text,
            "seconds": time.time() - tracker.start,
            "duration": len(speech) / SAMPLE_RATE,
        }

    # Render results
    res = st.session_state.get("result")
    if res and res["key"] == file_key:
        if not res["text"]:
            st.warning("No speech detected. Try speaking louder or closer to the microphone.")
        else:
            safe_text = html.escape(res["text"])
            st.markdown(
                f'<div class="result"><div class="label">✅ Transcription complete</div>'
                f'<div class="text">{safe_text}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Audio length {res['duration']:.1f}s · transcribed in {res['seconds']:.1f}s"
            )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download as .txt",
                    res["text"],
                    file_name="nuer_transcription.txt",
                )
            with col2:
                with st.expander("📋 Copy text"):
                    st.code(res["text"], language=None)
else:
    st.markdown(
        '<div class="hint">👆 Record audio or upload a file above to begin</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="footer">Powered by Meta MMS · adapter: {ADAPTER_REPO}</div>',
    unsafe_allow_html=True,
)
