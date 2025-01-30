import streamlit as st
import whisper
import pydub
import os
import requests
import re
from io import BytesIO
from tiktoken import get_encoding
import ssl
import json
from datetime import datetime
import time

# Temporary fix: Bypass SSL verification for Whisper model download
ssl._create_default_https_context = ssl._create_unverified_context

# Load Whisper model (optimized with a smaller model)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # Use the smaller 'base' model for faster transcription

# Function to convert audio file to WAV format if needed
def convert_audio_to_wav(input_file, output_file):
    audio = pydub.AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")

# Function to transcribe audio using Whisper
def transcribe_audio(file_path, model):
    result = model.transcribe(file_path, language='en', fp16=False)
    return result['text']

# Function to clean up the transcription text
def clean_transcription(transcription):
    transcription = re.sub(r'\b(uh|um|ah|like|you know|so)\b', '', transcription)
    transcription = re.sub(r'\s+', ' ', transcription)
    return transcription.strip()

# Function to split text into chunks based on token limit
def split_into_chunks(text, max_tokens=1500):
    encoding = get_encoding("cl100k_base")  # Use OpenAI's tokenizer for accurate token counting
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks

# Function to interact with Ollama's REST API (handles streaming responses)
def generate_mom_with_ollama(transcription):
    clean_text = clean_transcription(transcription)
    chunks = split_into_chunks(clean_text, max_tokens=1500)
    summaries = []

    for chunk in chunks:
        # Prepare the prompt
        prompt = f"""
        Generate a well-structured and organized Moments of Meeting (MoM) from the following meeting transcription. Ensure the output is concise, focuses on all key points, and highlights the following:
        1. **Action Items**: Clearly list all tasks, responsibilities, deadlines, and follow-ups discussed in the meeting.
        2. **Important Topics**: Summarize the main topics and decisions made during the meeting.
        3. **Key Takeaways**: Provide a brief summary of the most critical points discussed. Format the output in a clean and organized manner with appropriate headings and bullet points. Here is the transcription:\n{chunk}
        """

        # Send the request to the Ollama REST API
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "phi3", "prompt": prompt},
                stream=True  # Enable streaming
            )
            response.raise_for_status()  # Raise an error for HTTP issues

            # Process the streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:  # Skip empty lines
                    try:
                        # Parse each line as JSON
                        data = json.loads(line)
                        full_response += data.get("response", "")
                    except json.JSONDecodeError as e:
                        st.error(f"Error decoding JSON: {e}")
                        st.write("Raw Line:", line.decode("utf-8"))
                        return None

            summaries.append(full_response.strip())

        except requests.exceptions.RequestException as e:
            st.error(f"Error generating MoM with Ollama: {e}")
            return None

    # Combine all summaries
    full_summary = ' '.join(summaries)
    return full_summary

# Function to save text files locally with a timestamp
def save_to_local_folder(folder_name, file_name, content):
    # Ensure the folder exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Create the full file path
    file_path = os.path.join(folder_name, file_name)

    # Save the content to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path

# Function to load all files from a folder
def load_files_from_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        return []

    files = []
    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                files.append((file_name, f.read()))
    return files

# Streamlit UI
def main():
    st.title("Meeting Transcription and MoM Generator (Ollama Integration)")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Transcription", "MoM Generator"])

    # Initialize session state for storing transcripts and MoMs
    if "transcripts" not in st.session_state:
        st.session_state["transcripts"] = []
    if "moms" not in st.session_state:
        st.session_state["moms"] = []

    # Load previously saved transcriptions and MoMs
    saved_transcriptions = load_files_from_folder("transcripts")
    saved_moms = load_files_from_folder("moms")

    # Section 1: Transcription
    if page == "Transcription":
        st.header("Audio Transcription")
        uploaded_file = st.file_uploader("Upload an audio file (e.g., .mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])
        whisper_model = load_whisper_model()

        if uploaded_file is not None:
            start_time = time.time()
            with st.spinner("Processing audio file..."):
                # Save uploaded file temporarily
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Convert to WAV if necessary
                if not temp_file_path.lower().endswith('.wav'):
                    temp_wav_file = "temp_audio.wav"
                    convert_audio_to_wav(temp_file_path, temp_wav_file)
                    temp_file_path = temp_wav_file

                # Transcribe audio
                transcription = transcribe_audio(temp_file_path, whisper_model)

                # Clean up temporary files
                os.remove(temp_file_path)
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")

                end_time = time.time()
                st.success(f"Transcription completed in {end_time - start_time:.2f} seconds!")
                st.text_area("Transcription Output", transcription, height=200)

                # Save transcription
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"transcription_{timestamp}.txt"
                save_to_local_folder("transcripts", file_name, transcription)
                st.session_state["transcripts"].append(transcription)

                # Download button for transcription
                transcription_file = BytesIO()
                transcription_file.write(transcription.encode())
                transcription_file.seek(0)
                st.download_button(
                    label="Download Transcription",
                    data=transcription_file,
                    file_name=file_name,
                    mime="text/plain"
                )

        # Display saved transcripts
        st.subheader("Saved Transcripts")
        for file_name, content in saved_transcriptions:
            with st.expander(file_name):
                st.write(content)

    # Section 2: MoM Generator
    elif page == "MoM Generator":
        st.header("Moments of Meeting (MoM) Generator")
        uploaded_transcription = st.text_area("Paste or upload a transcription for MoM generation", height=200)
        uploaded_text_file = st.file_uploader("Upload a text file containing a transcription", type=["txt"])

        if uploaded_text_file is not None:
            uploaded_transcription = uploaded_text_file.read().decode("utf-8")
            st.text_area("Uploaded Transcription", uploaded_transcription, height=200)

        if st.button("Generate MoM"):
            if uploaded_transcription:
                start_time = time.time()
                with st.spinner("Generating MoM with Ollama..."):
                    mom = generate_mom_with_ollama(uploaded_transcription)
                    end_time = time.time()
                    if mom:
                        st.success(f"MoM generated successfully in {end_time - start_time:.2f} seconds!")
                        st.text_area("Generated MoM", mom, height=300)

                        # Save MoM
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file_name = f"mom_{timestamp}.txt"
                        save_to_local_folder("moms", file_name, mom)
                        st.session_state["moms"].append(mom)

                        # Download button for MoM
                        mom_file = BytesIO()
                        mom_file.write(mom.encode())
                        mom_file.seek(0)
                        st.download_button(
                            label="Download MoM",
                            data=mom_file,
                            file_name=file_name,
                            mime="text/plain"
                        )
            else:
                st.warning("Please provide a transcription for MoM generation.")

        # Display saved MoMs
        st.subheader("Saved MoMs")
        for file_name, content in saved_moms:
            with st.expander(file_name):
                st.write(content)

if __name__ == "__main__":
    main()