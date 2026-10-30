import os
import shutil
import subprocess
import tempfile

import imageio_ffmpeg
import librosa
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoProcessor, Wav2Vec2ForCTC

# App Configuration & Title
st.set_page_config(page_title="Nuer MMS Speech-to-Text", page_icon="🎙️")

st.title("🎙️ Nuer MMS Speech-to-Text")
st.write(
    "Upload a Nuer audio file to transcribe it using your fine-tuned MMS"
    " adapter model (`buayism/nuer_mms_adapter`)."
)

BASE_MODEL_ID = "facebook/mms-1b-all"
ADAPTER_REPO = "buayism/nuer_mms_adapter"
ADAPTER_FILE = "adapter.nus.safetensors"


# Load base model and adapter with caching so it only loads once
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


def load_audio_16k(uploaded_file):
    """Convert any uploaded audio to 16kHz mono WAV with ffmpeg, then load it."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(uploaded_file.getbuffer())
        src_path = src.name
    out_path = src_path + "_16k.wav"

    # Prefer system ffmpeg (packages.txt), fall back to the bundled binary
    ffmpeg_exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

    try:
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", src_path, "-ac", "1", "-ar", "16000", out_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1000:]}")
        speech_array, _ = librosa.load(out_path, sr=16000)
    finally:
        for p in (src_path, out_path):
            if os.path.exists(p):
                os.remove(p)

    return speech_array


with st.spinner("Initializing Nuer Speech-to-Text pipeline... Please wait."):
    processor, model = load_model()

# File uploader for audio files
audio_file = st.file_uploader(
    "Choose an audio file", type=["wav", "mp3", "m4a", "aac", "ogg"]
)

if audio_file is not None:
    st.audio(audio_file)

    if st.button("Transcribe Audio", type="primary"):
        with st.spinner("Processing speech-to-text transcription..."):
            try:
                speech_array = load_audio_16k(audio_file)
            except Exception as e:
                st.error(f"Could not read this audio file: {e}")
                st.stop()

            inputs = processor(
                speech_array, sampling_rate=16000, return_tensors="pt", padding=True
            )

            with torch.no_grad():
                logits = model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            st.success("Transcription Complete:")
            st.markdown(f"### {transcription}")
