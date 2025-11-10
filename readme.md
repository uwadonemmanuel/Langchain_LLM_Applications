# Langchain LLM - Streamlit Apps Launcher

This application provides an interactive interface to easily launch and manage multiple Streamlit applications, such as a Voice Assistant, News Summarizer, and YouTube Video Summarizer. It allows you to select an app, launch it on a specified port, and view its details from a centralized dashboard.

## 1. Overview

The Streamlit Apps Launcher is a tool to manage and launch a variety of Streamlit applications, each with unique functionalities. Users can quickly access these applications through a unified interface, with each app running in its own browser tab on different ports.

### Available Apps

- **Voice Assistant RAG**: A voice-powered retrieval-augmented generation (RAG) assistant that processes documents and supports text-to-speech (TTS) with both ElevenLabs and Deepgram integration.
- **News Summarizer**: Your friendly News Summarizer powered by OpenAI/Ollama models that Summarizes articles.
- **YouTube Summarizer**: Your friendly YouTube Video Summarizer that downloads and transcribes YouTube videos using Whisper and generates comprehensive summaries of video content.

## 2. Features

- **App Selection**: Choose from available Streamlit apps through an intuitive dropdown interface
- **Automatic Launching**: Launch apps directly from the Streamlit interface, opening them in a new browser tab on their respective ports
- **App Information**: View detailed information about each app, including its description, file name, and port number
- **Real-time Status**: Monitor whether each app is running, with the option to relaunch it if needed
- **Multiple Simultaneous Apps**: Launch and manage multiple apps at the same time on different ports
- **Port Management**: Each app runs on a dedicated port to avoid conflicts

## 3. Tools & Frameworks Used

- **Python**: The core programming language for building the application
- **Streamlit**: Used for creating the interactive web interface
- **Subprocess**: To manage launching apps in new processes
- **Webbrowser**: Automatically opens the apps in your browser after they are launched

## 4. Setup and Installation

### 4.1. Prerequisites

- Python 3.12 installed
- Streamlit installed (`pip install streamlit`)
- Basic understanding of Python and Streamlit

### 4.2. Clone the Repository

```bash
git clone https://github.com/Andela-GenAI/Langchain-LLM-Applications
cd langchain_llm_applications
```

### 4.3. Install Dependencies

If you are using a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate.bat  # On Windows
```

Install required packages:

```bash
pip install -r requirements.txt
```

NB: Ensure to set up ffmpeg location for yt_dlp and Whisper

### 4.4. Environment Variables

Create a `.env` file in the project root with your API keys:

```env
# Required for Voice Assistant RAG
OPENAI_API_KEY=your_openai_api_key_here
ELEVEN_LABS_API_KEY=your_elevenlabs_api_key_here
DEEPGRAM_API_KEY=your_deepgram_api_key_here

```

### 4.5. Running the App

To run the Streamlit Apps Launcher:

```bash
streamlit run launcher.py
```

This will open the launcher interface in your browser, where you can select an app and launch it.

## 5. Usage

### 5.1. Select and Launch Apps

1. **Choose an app**: In the sidebar, select one of the available apps from the dropdown
2. **Launch the app**: Click the "🚀 Launch App" button to start the selected app on its designated port
3. **View app info**: In the main panel, see the app's details, including a description, file name, and port number
4. **Manage running apps**: If the app is already running, you can click "🔄 Relaunch" to restart it

### 5.2. Access Running Apps

Once an app is launched, you will be able to access it at `http://localhost:{port}`. Each app runs on a different port:

For example:
- **Voice Assistant RAG**: `http://localhost:8502`
- **Lunar News Summarizer**: `http://localhost:8503`

The launcher automatically opens the app in a new browser tab when launched.

### 5.3. Running Multiple Apps

You can launch and run multiple apps simultaneously. Each app runs on its own port, so there are no conflicts. The launcher tracks which apps are currently running and displays their status.

## 6. Project Structure

```
.
├── launcher.py                      # Streamlit Apps Launcher
├── voice_assis3.py              # Voice Assistant RAG App
├── streamlit_app_reduced.py    # Lunar AI Assistant App
├── setup_chromadb.py           # ChromaDB setup script
├── new_requirements.txt       # Minimal dependencies
├── requirements.txt            # Full dependencies
└── README.md                   # Project documentation
```

## 7. App Details

### Voice Assistant RAG (`voice_assist_streamlit.py`)

Lunar - A comprehensive voice-powered RAG system that:

- Processes documents (PDF, TXT, MD) and creates a vector store
- Records audio input using Whisper for transcription
- Generates responses using RAG with conversational memory
- Converts text to speech using ElevenLabs or Deepgram
- Supports multiple TTS providers with voice selection

**Port**: 8502

### News Summarizer (`news_summarizer_streamlit.py`)

Lunar - Your friendly News Summarizer that:

- Extracts and processes news articles from URLs
- Summarizes articles using OpenAI or Ollama models
- Provides concise summaries with key points
- Supports multiple news sources and article formats
- Offers customizable summarization length and style

**Port**: 8503

### YouTube Summarizer (`yt_vid_summarizer_streamlit.py`)

Lunar - Your friendly YouTube Video Summarizer that:

- Downloads and transcribes YouTube videos using Whisper
- Extracts key information from video transcripts
- Generates comprehensive summaries of video content
- Supports RAG-based question answering about video content
- Provides conversational interface for video queries

**Port**: 8504

## 8. Extendability

This launcher provides a foundation for managing and launching Streamlit apps. Here are some ideas for extending its functionality:

- **Add more apps**: Integrate additional Streamlit apps into the launcher by updating the `AVAILABLE_APPS` dictionary in `launcher.py`
- **App Configuration**: Provide customizable settings for each app, such as input options or model configuration
- **Improved UI**: Enhance the user interface with custom themes and more interactive components
- **App Monitoring**: Display logs or stats for each running app in real-time
- **App Management**: Add functionality to stop/kill running apps from the launcher
- **Port Auto-assignment**: Automatically assign available ports instead of hardcoding them

### Adding a New App

To add a new app to the launcher, update the `AVAILABLE_APPS` dictionary in `launcher.py`:

```python
AVAILABLE_APPS = {
    "Your App Name": {
        "file": "your_app.py",
        "description": "Description of your app",
        "icon": "🎯",
        "port": 8504  # Use a different port
    }
}
```

## 9. Troubleshooting

### Port Already in Use

If you encounter a "port already in use" error:

1. Check which process is using the port: `lsof -i :8502` (macOS/Linux) or `netstat -ano | findstr :8502` (Windows)
2. Kill the process or choose a different port in the `AVAILABLE_APPS` configuration

### App Not Launching

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that the app file exists in the project directory
- Verify that required API keys are set in your `.env` file
- Check the terminal/console for error messages

### Browser Not Opening Automatically

- Manually navigate to the URL shown in the launcher (e.g., `http://localhost:8502`)
- Check the terminal output for the exact URL
- Ensure your browser allows pop-ups from localhost

## 10. License

This project is licensed under the MIT License. See the LICENSE file for more details.

## 11. Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 12. Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Note**: Make sure to set up your API keys in the `.env` file before running the applications. Some features may require specific API keys to function properly.

