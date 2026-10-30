import html
import os
import shutil
import subprocess
import tempfile
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
  animation: fadeUp .3s ease both;
}
.bars { display:flex; gap:5px; justify-content:center; align-items:center; height:50px; }
.bars span { width:6px; height:8px; background:#ff4b4b; border-radius:3px; animation: bounce 1s ease-in-out infinite; }
.bars span:nth-child(2){animation-delay:.1s} .bars span:nth-child(3){animation-delay:.2s}
.bars span:nth-child(4){animation-delay:.3s} .bars span:nth-child(5){animation-delay:.4s}
.bars span:nth-child(6){animation-delay:.3s} .bars span:nth-child(7){animation-delay:.2s}
.bars span:nth-child(8){animation-delay:.1s} .bars span:nth-child(9){animation-delay:0s}
.loader .msg { margin-top:.6rem; font-weight:600; }

.result {
  background: rgba(33,195,84,.1); border: 1px solid rgba(61,213,109,.5);
  border-radius: 16px; padding: 1.2rem 1.4rem; margin-top: 1rem;
  box-shadow: 0 0 24px rgba(33,195,84,.15); animation: fadeUp .5s ease both;
}
.result .label { color:#3dd56d; font-size:.8rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.result .text { font-size: 1.6rem; font-weight: 700; line-height: 1.5; margin-top:.4rem; }

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


# ---------- Helpers ----------
def loader_html(message):
    return (
        '<div class="loader"><div class="bars">'
        + "<span></span>" * 9
        + f'</div><div class="msg">{html.escape(message)}</div></div>'
    )


def load_audio_16k(uploaded_file):
    """Convert any uploaded audio to 16kHz mono WAV with ffmpeg, then load it."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(uploaded_file.getbuffer())
        src_path = src.name
    out_path = src_path + "_16k.wav"

    ffmpeg_exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

    try:
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", src_path, "-ac", "1", "-ar", str(SAMPLE_RATE), out_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-500:]}")
        speech_array, _ = librosa.load(out_path, sr=SAMPLE_RATE)
    finally:
        for p in (src_path, out_path):
            if os.path.exists(p):
                os.remove(p)

    return speech_array


def transcribe(speech, on_progress):
    """Transcribe audio; long recordings are split into 20s parts to save memory."""
    chunk = 20 * SAMPLE_RATE if len(speech) > 30 * SAMPLE_RATE else len(speech)
    pieces = [speech[i : i + chunk] for i in range(0, len(speech), chunk)]
    if len(pieces) > 1 and len(pieces[-1]) < SAMPLE_RATE:
        pieces.pop()  # drop a tiny trailing sliver

    texts = []
    for idx, piece in enumerate(pieces, 1):
        on_progress(idx, len(pieces))
        inputs = processor(
            piece, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        with torch.inference_mode():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
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
        loader = st.empty()
        start = time.time()
        try:
            loader.markdown(
                loader_html("Preparing your audio…"), unsafe_allow_html=True
            )
            speech = load_audio_16k(audio_file)
            if len(speech) == 0:
                raise RuntimeError("The audio file appears to be empty.")

            def show_progress(i, n):
                msg = (
                    "Listening to your recording…"
                    if n == 1
                    else f"Listening… part {i} of {n}"
                )
                loader.markdown(loader_html(msg), unsafe_allow_html=True)

            text = transcribe(speech, show_progress)
        except Exception as e:
            loader.empty()
            st.error(f"Something went wrong: {e}")
            st.stop()

        loader.empty()
        st.session_state["result"] = {
            "key": file_key,
            "text": text,
            "seconds": time.time() - start,
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
