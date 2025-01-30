# Meeting Transcription and MoM Generator

An AI-powered application that transcribes audio recordings and automatically generates Minutes of Meeting (MoM) using Whisper ASR and Ollama LLM.

## Features

- 🎙️ Audio transcription supporting multiple formats (.mp3, .wav, .m4a)
- 📝 Automatic Minutes of Meeting generation
- 💾 Local storage for transcripts and MoMs
- ⚡ Stream processing for handling long recordings
- 🔄 Clean text processing and formatting
- 📥 Download options for both transcripts and MoMs

## Tech Stack

- **Streamlit**: Web interface
- **Whisper**: OpenAI's speech recognition model
- **Ollama**: Local LLM for MoM generation
- **Pydub**: Audio file processing
- **TikToken**: Token management for LLM input

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running locally
3. Sufficient disk space for model storage

## Installation

```bash
# Clone the repository
git clone https://github.com/Rishurajgautam24/AI_powered_MOM.git
cd AI_powered_MOM

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running locally with the phi3 model
ollama pull phi3
```

## Usage

1. Start the application:
```bash
streamlit run ASR.py
```

2. Navigate to either:
   - **Transcription**: Upload audio files for transcription
   - **MoM Generator**: Generate meeting minutes from transcripts

3. The application will automatically:
   - Save transcriptions to a `transcripts` folder
   - Save generated MoMs to a `moms` folder

## Project Structure

```
├── ASR.py              # Main application file
├── transcripts/        # Stored transcriptions
├── moms/              # Stored Minutes of Meetings
└── requirements.txt    # Project dependencies
```

## Features in Detail

### Transcription
- Supports multiple audio formats
- Automatic format conversion to WAV when needed
- Uses Whisper's base model for efficient processing
- Cleans and formats transcribed text

### MoM Generation
- Processes long transcripts in chunks
- Generates structured meeting minutes including:
  - Action Items
  - Important Topics
  - Key Takeaways
- Handles streaming responses from Ollama

## Performance Considerations

- Uses Whisper's base model for faster transcription
- Implements token-based chunking for long texts
- Caches the Whisper model for improved performance
- Streams Ollama responses for better memory management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- OpenAI's Whisper for ASR capabilities
- Ollama for local LLM processing
- Streamlit for the web interface
