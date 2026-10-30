import librosa
import streamlit as st
import torch
from transformers import AutoModelForCTC, AutoProcessor

# App Configuration & Title
st.set_page_config(page_title="Nuer MMS Speech-to-Text", page_icon="🎙️")

st.title("🎙️ Nuer MMS Speech-to-Text")
st.write(
    "Upload a Nuer audio file or record your voice to transcribe it using your"
    " fine-tuned MMS adapter model (`buayism/nuer_mms_adapter`)."
)


# Load model and processor with caching so it only loads once
@st.cache_resource
def load_model():
  model_id = "buayism/nuer_mms_adapter"
  processor = AutoProcessor.from_pretrained(model_id)
  model = AutoModelForCTC.from_pretrained(model_id)
  return processor, model


with st.spinner(
    "Loading Nuer MMS model and adapter from Hugging Face... Please wait."
):
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