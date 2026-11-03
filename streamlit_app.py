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
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2ForCTC

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

st.set_page_config(
    page_title="Nuer MMS Speech-to-Text", page_icon="🎙️", layout="centered"
)

BASE_MODEL_ID = "facebook/mms-1b-all"
ADAPTER_REPO = "buayism/nuer_mms_adapter"
ADAPTER_FILE = "adapter.nus.safetensors"
SAMPLE_RATE = 16000
TOTAL_STEPS = 6
BAR_DELAYS = [0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0]

# ---------- Styling (Enlarged Audio Input & Clean UI) ----------
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

/* --- ENLARGE STREAMLIT NATIVE AUDIO INPUT CONTROLS --- */
[data-testid="stAudioInput"] {
    background-color: #1e1f29;
    border: 2px solid rgba(255, 75, 75, 0.4);
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
/* Increase record/play buttons and icons inside audio input */
[data-testid="stAudioInput"] button {
    transform: scale(1.4);
    margin: 0.4rem 0.6rem;
}
/* Enlarge timer text and waveform visuals */
[data-testid="stAudioInput"] svg {
    transform: scale(1.2);
}
[data-testid="stAudioInput"] span, [data-testid="stAudioInput"] div {
    font-size: 1.1rem !important;
}

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
<div class="sub">Turn spoken Nuer into text.</div>
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
# Retrieve Hugging Face token from Streamlit secrets (with fallback to environment variables)
hf_token = None
try:
    if "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]
except Exception:
    # Occurs if secrets are not configured or initialized locally
    pass

if not hf_token:
    hf_token = os.environ.get("HF_TOKEN")


@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def load_model(token: str | None = None):
    # Load processor directly from the adapter repo
    processor = AutoProcessor.from_pretrained(
        ADAPTER_REPO,
        token=token,
    )

    # Load base model
    model = AutoModelForCTC.from_pretrained(
        BASE_MODEL_ID,
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
        token=token,
    )

    # Download adapter weights using private repository token
    adapter_path = hf_hub_download(
        repo_id=ADAPTER_REPO,
        filename=ADAPTER_FILE,
        token=token,
    )
    state_dict = load_file(adapter_path)
    
    # Load state dict directly without manual embedding resizing
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.eval()
    return processor, model


# Fallback warning if token is missing
if not hf_token:
    st.warning(
        "⚠️ **Hugging Face token (`HF_TOKEN`) not detected in Streamlit secrets.**\n\n"
        "If your adapter repository is private, loading will fail. "
        "Please add `HF_TOKEN = \"your_token\"` to your Streamlit secrets."
    )

try:
    with st.spinner("Warming up the Nuer model..."):
        processor, model = load_model(token=hf_token)
except Exception as e:
    err_str = str(e)
    if any(keyword in err_str.lower() for keyword in ["401", "403", "unauthorized", "gated", "restricted", "token", "not found"]):
        st.error(
            "🔒 **Hugging Face Authentication / Access Error**: Unable to access the adapter repository "
            f"`{ADAPTER_REPO}`.\n\n"
            "The repository is private or the provided token is unauthorized or missing. "
            "Please make sure `HF_TOKEN` in your Streamlit secrets is valid and has read access."
        )
    else:
        st.error(f"⚠️ **Error loading model**: {err_str}")

    st.info(
        "💡 **How to configure your token in Streamlit Cloud:**\n"
        "1. Open your app on [share.streamlit.io](https://share.streamlit.io).\n"
        "2. Click **Settings** (⋮) → **Secrets**.\n"
        "3. Add your Hugging Face Access Token:\n"
        "```toml\n"
        'HF_TOKEN = "hf_your_token_here"\n'
        "```"
    )
    st.stop()


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


# ---------- Main UI Layout (Tabs for Live Recording vs File Upload) ----------
tab_record, tab_upload = st.tabs(["🎙️ Record Live Audio", "📁 Upload File"])

audio_bytes = None
file_key = None

with tab_record:
    st.caption("Click the microphone button to record your voice directly. Once stopped, you can play it back.")
    recorded_file = st.audio_input("Record Nuer Voice")
    
    if recorded_file is not None:
        audio_bytes = recorded_file.read()
        file_key = f"live_recording_{len(audio_bytes)}"

with tab_upload:
    uploaded_file = st.file_uploader(
        "Drop a recording here or browse",
        type=["wav", "mp3", "m4a", "aac", "ogg"],
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
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

    # Render results & clear option
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
            
            col_dl, col_copy = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️ Download as .txt",
                    res["text"],
                    file_name="nuer_transcription.txt",
                )
            with col_copy:
                with st.expander("📋 Copy text"):
                    st.code(res["text"], language=None)

            # Clear / Reset Button to record another audio immediately
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Record New Audio / Clear Screen"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
else:
    st.markdown(
        '<div class="hint">👆 Record audio or upload a file above to begin</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="footer">Powered by Meta MMS · adapter: {ADAPTER_REPO}</div>',
    unsafe_allow_html=True,
)