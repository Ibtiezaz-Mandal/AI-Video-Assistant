# 🎥 AI Video Assistant

An AI-powered video and meeting assistant that transforms YouTube videos and local video files into searchable, interactive knowledge sources.

The application automatically extracts audio, generates speech transcriptions, creates AI-powered summaries, identifies key decisions and action items, and enables users to ask questions about video content through a Retrieval-Augmented Generation (RAG) system.

## ✨ Features

- **YouTube Video Processing:** Extract audio from YouTube videos using `yt-dlp`.
- **Local Video Processing:** Process locally stored video files.
- **Speech-to-Text Transcription:** Generate transcripts using OpenAI Whisper.
- **Multilingual Transcription:** Support for English and Hindi transcription using Whisper and Sarvam AI.
- **AI-Powered Summarization:** Generate concise and meaningful video summaries using Google Gemini.
- **Key Point Extraction:** Identify important topics and information discussed in videos.
- **Action Item Extraction:** Extract tasks and actionable points from video content.
- **Decision Identification:** Highlight important decisions made during meetings or discussions.
- **RAG-Based Question Answering:** Ask questions about video content and receive context-aware answers.
- **Vector Database:** Store and retrieve transcript embeddings using ChromaDB.
- **Interactive Web Interface:** User-friendly interface built with Streamlit.
- **Export Functionality:** Export generated results in PDF and TXT formats.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| OpenAI Whisper | Speech recognition and transcription |
| Sarvam AI | Hindi speech-to-text processing |
| Google Gemini | AI-powered summarization and responses |
| LangChain | LLM integration and RAG orchestration |
| ChromaDB | Vector database for semantic search |
| Hugging Face | Text embedding models |
| yt-dlp | YouTube audio extraction |
| Pydub | Audio processing and segmentation |

## 🏗️ Project Architecture

The application follows a modular pipeline:

1. **Input:** Accept a YouTube URL or a local video file.
2. **Audio Extraction:** Extract audio from the video using `yt-dlp` or the local video processing pipeline.
3. **Audio Processing:** Process and divide audio into manageable segments.
4. **Transcription:** Convert speech into text using Whisper or Sarvam AI.
5. **Summarization:** Generate summaries, key points, decisions and action items using Google Gemini.
6. **Vector Storage:** Generate embeddings and store transcript chunks in ChromaDB.
7. **Question Answering:** Retrieve relevant transcript chunks and generate context-aware answers using the RAG pipeline.
8. **Output:** Display results through the Streamlit interface and provide export options.

## 📂 Project Structure

```text
AI-Video-Assistant/
│
├── core/
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarize.py
│   ├── transcriber.py
│   └── vector_store.py
│
├── utils/
│   └── audio_processor.py
│
├── .gitignore
├── README.md
├── Requirements.txt
├── app.py
└── main.py
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12
- Git
- FFmpeg
- Google Gemini API key
- Sarvam AI API key (for Hindi transcription)

### 1. Clone the Repository

```bash
git clone https://github.com/Ibtiezaz-Mandal/AI-Video-Assistant.git
```

Navigate to the project directory:

```bash
cd AI-Video-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r Requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory.

Add your API keys:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

Replace the placeholder values with your actual API keys.

**Important:** Never upload your `.env` file or expose your API keys on GitHub.

### 5. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be accessible through your local browser.

## 💡 How to Use

1. Launch the Streamlit application.
2. Provide a YouTube video URL or upload a local video.
3. Select the appropriate transcription language.
4. Start the processing pipeline.
5. View the generated transcript and AI-powered summary.
6. Explore key points, decisions and action items.
7. Ask questions about the video using the RAG-based chat interface.
8. Export the results in PDF or TXT format.

## 🔮 Future Improvements

- Support for additional languages.
- Improved transcription accuracy and processing speed.
- Real-time meeting transcription and summarization.
- Speaker identification and diarization.
- Enhanced conversational memory for multi-turn RAG interactions.
- Cloud deployment for public access.
- Support for additional video platforms.

## 👨‍💻 Author

**Ibtiezaz Mandal**

Master of Computer Applications (MCA)  
National Institute of Technology, Patna

## 📄 License

This project is currently available for educational and learning purposes.
