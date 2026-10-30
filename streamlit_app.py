import os
import tempfile

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
    # 1. Processor/tokenizer with your Nuer vocab
    processor = AutoProcessor.from_pretrained(ADAPTER_REPO)

    # 2. Base MMS model, resized to your Nuer vocab size
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL_ID,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    # 3. Load your trained adapter layers + lm_head from your HF repo
    adapter_path = hf_hub_download(ADAPTER_REPO, ADAPTER_FILE)
    state_dict = load_file(adapter_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Unexpected adapter keys:", unexpected)  # should be empty

    model.eval()
    return processor, model


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
            # Save the upload temporarily, keeping its real file extension
            suffix = os.path.splitext(audio_file.name)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.getbuffer())
                audio_path = tmp.name

            try:
                # Load and resample audio to 16kHz (required by MMS model)
                speech_array, _ = librosa.load(audio_path, sr=16000)
            finally:
                os.remove(audio_path)

            # Prepare model inputs
            inputs = processor(
                speech_array, sampling_rate=16000, return_tensors="pt", padding=True
            )

            # Run inference without gradient calculations
            with torch.no_grad():
                logits = model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            # Display the final transcription
            st.success("Transcription Complete:")
            st.markdown(f"### {transcription}")
