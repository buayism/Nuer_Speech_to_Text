import html
import os
import shutil
import subprocess
import tempfile
import threading
import time

import imageio_ffmpeg
import librosa
import streamlit as st
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

# ---------- Styling (same palette: dark bg, red #ff4b4b, green) ----------
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 760px; }

@keyframes fadeUp { from {opacity:0; transform: translateY(14px);} to {opacity:1; transform:none;} }
@keyframes pulse {
  0%,100% { transform: scale(1); filter: drop-shadow(0 0 0 rgba(255,75,75,0)); }
  50% { transform: scale(1.1); filter: drop-shadow(0 0 14px rgba(255,75,75,.75)); }
}
@keyframes bounce { 0%,100% { height: 8px; } 50% { height: 44px; } }

.hero { text-align:center; animation: fadeUp .6s ease both; }
.hero .mic { font-size: 3.4rem; display:inline-block; animation: pulse 2.4s ease-in-out infinite; }
.hero .title {
  font-size: 2.5rem; font-weight: 800; line-height: 1.15; margin: .3rem 0 .4rem;
  background: linear-gradient(90deg, #ff4b4b, #ffffff);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero .sub { opacity: .8; margin-bottom: .4rem; }

.steps { display:flex; gap:.6rem; justify-content:center; flex-wrap:wrap; margin: 1rem 0 1.6rem; animation: fadeUp .8s ease both; }
.step { padding:.4rem .95rem; border-radius:999px; background:#262730; font-size:.85rem; border:1px solid rgba(255,75,75,.35); }
.step b { color:#ff4b4b; margin-right:.3rem; }

[data-testid="stFileUploader"] section {
  border: 2px dashed rgba(255,75,75,.55); border-radius: 16px; background: #262730;
  padding: 1.4rem; transition: all .25s ease;
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

/* Copy box uses the same Times New Roman style */
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
<div class="step"><b>1</b>Upload audio</div>
<div class="step"><b>2</b>Click transcribe</div>
<div class="step"><b>3</b>Copy or download</div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Model ----------
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
    print("Unexpected adapter keys:", unexpected)  # should be empty

    model.eval()
    return processor, model


with st.spinner("Warming up the Nuer model (the first load can take a minute)..."):
    processor, model = load_model()


# ---------- Live, honest progress display ----------
def loader_html(step, title, note, elapsed, progress=None):
    # Negative delays keep the bar animation in phase when the card re-renders
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
    """Writes the current, real pipeline stage into one placeholder."""

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.start = time.time()

    def show(self, step, title, note, progress=None):
        self.placeholder.markdown(
            loader_html(step, title, note, time.time() - self.start, progress),
            unsafe_allow_html=True,
        )


# ---------- Audio + model helpers ----------
def load_audio_16k(uploaded_file, tracker):
    """Convert any uploaded audio to 16kHz mono WAV with ffmpeg, then load it."""
    tracker.show(1, "Reading your audio file…", "Saving your upload so it can be processed.")
    suffix = os.path.splitext(uploaded_file.name)[1] or ".audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(uploaded_file.getbuffer())
        src_path = src.name
    out_path = src_path + "_16k.wav"

    ffmpeg_exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

    try:
        tracker.show(
            2,
            "Converting your audio…",
            "Converting to 16 kHz mono, the format the model was trained to hear.",
        )
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", src_path, "-ac", "1", "-ar", str(SAMPLE_RATE), out_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-500:]}")

        tracker.show(
            2,
            "Loading the audio into memory…",
            "Reading the converted sound samples so the model can use them.",
        )
        speech_array, _ = librosa.load(out_path, sr=SAMPLE_RATE)
    finally:
        for p in (src_path, out_path):
            if os.path.exists(p):
                os.remove(p)

    return speech_array


def attach_progress_hooks(state):
    """Real progress signals from inside the model's forward pass."""
    w2v = model.wav2vec2
    handles = [
        # Acoustic feature extraction finished -> transformer layers start
        w2v.feature_projection.register_forward_hook(
            lambda m, a, o: state.update(stage="encoder")
        ),
        # Prediction head is about to run -> all transformer layers are done
        model.lm_head.register_forward_pre_hook(
            lambda m, a: state.update(stage="head")
        ),
    ]
    # Count each transformer layer as it actually completes
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
    except Exception as e:  # surfaced to the main thread
        result["error"] = e


def transcribe(speech, tracker):
    """Transcribe audio; long recordings are split into 20s parts to save memory."""
    chunk = 20 * SAMPLE_RATE if len(speech) > 30 * SAMPLE_RATE else len(speech)
    pieces = [speech[i : i + chunk] for i in range(0, len(speech), chunk)]
    if len(pieces) > 1 and len(pieces[-1]) < SAMPLE_RATE:
        pieces.pop()  # drop a tiny trailing sliver

    total_layers = len(model.wav2vec2.encoder.layers)
    texts = []

    for idx, piece in enumerate(pieces, 1):
        part = f"Part {idx} of {len(pieces)} · " if len(pieces) > 1 else ""

        tracker.show(
            3,
            "Preparing the audio for the model…",
            part + "Normalizing the volume and shaping the data into what the model expects.",
        )
        inputs = processor(
            piece, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )

        # Run the model in a background thread so this page can report real progress
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
                        part + "Turning the raw sound wave into patterns the model can analyze.",
                    )
                elif stage == "encoder":
                    done = min(state["layer"], total_layers)
                    tracker.show(
                        4,
                        "Analyzing the speech…",
                        f"{part}Neural network layer {done} of {total_layers}. "
                        "This is the most computation-heavy step.",
                        progress=done / total_layers,
                    )
                else:
                    tracker.show(
                        5,
                        "Predicting Nuer characters…",
                        part + "Scoring every possible character for each moment of audio.",
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
            "Decoding predictions into text…",
            part + "Last step: choosing the best character at each moment and joining them into words.",
        )
        predicted_ids = torch.argmax(result["logits"], dim=-1)
        texts.append(processor.batch_decode(predicted_ids)[0].strip())

    return " ".join(t for t in texts if t)


# ---------- Main UI ----------
audio_file = st.file_uploader(
    "Drop a recording here or browse",
    type=["wav", "mp3", "m4a", "aac", "ogg"],
)

with st.expander("💡 Tips for best results"):
    st.write(
        "- Speak clearly and keep background noise low\n"
        "- Hold the microphone close to the speaker\n"
        "- Short clips (under 30 seconds) are the fastest\n"
        "- Supported formats: WAV, MP3, M4A, AAC, OGG"
    )

if audio_file is None:
    st.markdown(
        '<div class="hint">👆 Upload a Nuer recording to get started</div>',
        unsafe_allow_html=True,
    )
else:
    st.audio(audio_file)
    file_key = f"{audio_file.name}-{audio_file.size}"

    if st.button("✨ Transcribe Audio", type="primary"):
        placeholder = st.empty()
        tracker = Tracker(placeholder)
        try:
            speech = load_audio_16k(audio_file, tracker)
            if len(speech) == 0:
                raise RuntimeError("The audio file appears to be empty.")
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

    # Result is stored in session_state so it survives the download button's rerun
    res = st.session_state.get("result")
    if res and res["key"] == file_key:
        if not res["text"]:
            st.warning("No speech detected. Try a clearer or louder recording.")
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

st.markdown(
    f'<div class="footer">Powered by Meta MMS · adapter: {ADAPTER_REPO}</div>',
    unsafe_allow_html=True,
)
