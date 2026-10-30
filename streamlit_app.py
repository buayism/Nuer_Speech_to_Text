import librosa
import streamlit as st
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

# App Configuration & Title
st.set_page_config(page_title="Nuer MMS Speech-to-Text", page_icon="🎙️")

st.title("🎙️ Nuer MMS Speech-to-Text")
st.write(
    "Upload a Nuer audio file or record your voice to transcribe it using your"
    " fine-tuned MMS adapter model (`buayism/nuer_mms_adapter`)."
)


# Load base model and adapter with caching so it only loads once
@st.cache_resource
def load_model():
  # 1. Use the official base MMS model architecture
  base_model_id = (
      "facebook/mms-1b-all"  # or "facebook/mms-300m" depending on what you trained
  )
  adapter_repo = "buayism/nuer_mms_adapter"

  st.info("Loading base MMS model and processor...")
  processor = AutoProcessor.from_pretrained(adapter_repo)
  model = Wav2Vec2ForCTC.from_pretrained(base_model_id)

  # 2. Load your custom Nuer adapter weights from your Hugging Face repo
  st.info("Loading Nuer adapter weights...")
  model.load_adapter(adapter_repo)

  # 3. Set the active adapter language/token
  model.active_adapter = "nus"

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
      # Save the uploaded file locally temporarily
      audio_path = "temp_audio.wav"
      with open(audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

      # Load and resample audio to 16kHz (required by MMS model)
      speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

      # Tokenize input values
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
