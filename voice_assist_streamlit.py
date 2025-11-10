import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
from elevenlabs.client import ElevenLabs
import requests

from appconfig import env_config
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

import tempfile
from dotenv import load_dotenv
import os
from typing import List
from langchain_core.documents import Document

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings()

    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from different file types"""
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(
                directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
            ),
        }

        documents = []
        for file_type, loader in loaders.items():
            try:
                documents.extend(loader.load())
                print(f"Loaded {file_type} documents")
            except Exception as e:
                print(f"Error loading {file_type} documents: {str(e)}")

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)

    def create_vector_store(
        self, documents: List[Document], persist_directory: str
    ) -> Chroma:
        """Create and persist vector store if it doesn't exist, otherwise load existing one"""
        # Check if persist_directory exists and has content
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print(f"Loading existing vector store from {persist_directory}")
            # Load existing vector store
            vector_store = Chroma(
                persist_directory=persist_directory, embedding_function=self.embeddings
            )
        else:
            print(f"Creating new vector store in {persist_directory}")
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)

            # Create new vector store
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory,
            )
            vector_store.persist()

        return vector_store
    
class VoiceGenerator:
    def __init__(self, elevenlabs_api_key=None, deepgram_api_key=None, provider="elevenlabs"):
        self.provider = provider
        self.elevenlabs_api_key = elevenlabs_api_key
        self.deepgram_api_key = deepgram_api_key
        
        # Initialize ElevenLabs client if API key provided
        if elevenlabs_api_key:
            self.client = ElevenLabs(api_key=elevenlabs_api_key)
        else:
            self.client = None
        # Map voice names to their IDs (ElevenLabs uses IDs, not names)
        # Common voice IDs - you may need to update these or fetch from API
        self.voice_map = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",  # Rachel voice ID
            "Domi": "AZnzlk1XvdvUeBnXmlld",     # Domi voice ID
            "Bella": "EXAVITQu4vr4xnSDxMaL",    # Bella voice ID
            "Antoni": "ErXwobaYiN019PkySvjV",   # Antoni voice ID
            "Elli": "MF3mGyEYCl7XYWbV9V6O",     # Elli voice ID
            "Josh": "TxGEqnHWrfWFTfGW9XjX",     # Josh voice ID
            "Arnold": "VR6AewLTigWG4xSOukaG",   # Arnold voice ID
            "Adam": "pNInz6obpgDQGcFmaJgB",     # Adam voice ID
            "Sam": "yoZ06aMxZJJ28mfd3POQ",      # Sam voice ID
        }
        # Default available voices (names for UI)
        self.available_voices = list(self.voice_map.keys())
        self.default_voice = "Rachel"
        
        # Deepgram voices
        self.deepgram_voices = [
            "aura-asteria-en",
            "aura-luna-en",
            "aura-stella-en",
            "aura-athena-en",
            "aura-hera-en",
            "aura-orion-en",
            "aura-arcas-en",
            "aura-perseus-en",
            "aura-angus-en",
            "aura-orpheus-en",
            "aura-zeus-en",
            "aura-odin-en"
        ]
        
        # Try to fetch voices from API to get actual IDs
        if self.client:
            try:
                voices = self.client.voices.get_all()
                # Update voice_map with actual voice IDs from API
                for voice in voices.voices:
                    if voice.name in self.voice_map:
                        self.voice_map[voice.name] = voice.voice_id
            except Exception as e:
                print(f"Could not fetch voices from API, using defaults: {e}")

    def _generate_with_deepgram(self, text: str, voice: str = "aura-asteria-en") -> str:
        """Generate voice using Deepgram TTS API"""
        try:
            if not self.deepgram_api_key:
                raise RuntimeError("DEEPGRAM_API_KEY not set in environment variables.")
            
            # Deepgram TTS endpoint
            url = f"https://api.deepgram.com/v1/speak?model={voice}"
            headers = {
                "Authorization": f"Token {self.deepgram_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {"text": text}
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(response.content)
                return temp_audio.name
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise RuntimeError("Deepgram API key is invalid or expired.")
            elif e.response.status_code == 403:
                raise RuntimeError("Deepgram API key does not have TTS permissions or account lacks credits.")
            else:
                error_text = e.response.text[:200] if hasattr(e.response, 'text') else str(e)
                raise RuntimeError(f"Deepgram API error ({e.response.status_code}): {error_text}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Deepgram network error: {e}")
        except Exception as e:
            raise RuntimeError(f"Deepgram TTS error: {e}")
    
    def generate_voice_response(self, text: str, voice_name: str = None, provider: str = None) -> str:
        """Generate voice response with support for multiple providers"""
        selected_provider = provider or self.provider
        
        # Use Deepgram if selected
        if selected_provider == "deepgram":
            deepgram_voice = voice_name or "aura-asteria-en"
            return self._generate_with_deepgram(text, deepgram_voice)
        
        # Default to ElevenLabs
        if not self.client:
            raise RuntimeError("ElevenLabs API key not provided. Cannot generate voice.")
        
        selected_voice_name = voice_name or self.default_voice
        # Convert voice name to voice ID
        voice_id = self.voice_map.get(selected_voice_name, self.voice_map[self.default_voice])
        
        try:
            # Try the standard ElevenLabs API method
            # Check what's available on the client
            if hasattr(self.client, 'generate'):
                # Older API: direct generate method
                audio_generator = self.client.generate(
                    text=text, 
                    voice=voice_id, 
                    model="eleven_multilingual_v2"
                )
            elif hasattr(self.client, 'text_to_speech'):
                # Newer API: text_to_speech submodule
                if hasattr(self.client.text_to_speech, 'convert'):
                    # Use convert method (synchronous)
                    audio_data = self.client.text_to_speech.convert(
                        voice_id=voice_id,
                        text=text,
                        model_id="eleven_multilingual_v2"
                    )
                    # Handle the response - it might be bytes or have an audio attribute
                    if isinstance(audio_data, bytes):
                        audio_bytes = audio_data
                    elif hasattr(audio_data, 'audio'):
                        audio_bytes = audio_data.audio
                    else:
                        audio_bytes = bytes(audio_data)
                elif hasattr(self.client.text_to_speech, 'generate'):
                    # Use generate method
                    audio_generator = self.client.text_to_speech.generate(
                        voice_id=voice_id,
                        text=text,
                        model_id="eleven_multilingual_v2"
                    )
                else:
                    raise AttributeError("text_to_speech has no convert or generate method")
            else:
                raise AttributeError("Client has no generate or text_to_speech attribute")
            
            # Handle generator/iterator if that's what we got
            if 'audio_generator' in locals():
                audio_bytes = b"".join(audio_generator)
            elif 'audio_bytes' not in locals():
                raise ValueError("Could not extract audio bytes")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                return temp_audio.name
                
        except Exception as e:
            error_str = str(e)
            
            # Try to extract error details from the exception object
            status_code = None
            error_body = None
            
            # Check if error has status_code attribute (common in API clients)
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
                if hasattr(e.response, 'json'):
                    try:
                        error_body = e.response.json()
                    except:
                        pass
            
            # Check for 401 errors
            if status_code == 401 or "401" in error_str or "status_code: 401" in error_str:
                # Check for specific error messages
                if error_body and isinstance(error_body, dict):
                    detail = error_body.get('detail', {})
                    if isinstance(detail, dict):
                        message = detail.get('message', '')
                        status = detail.get('status', '')
                        
                        if 'detected_unusual_activity' in str(status) or 'Free Tier usage disabled' in message:
                            error_msg = (
                                "⚠️ ElevenLabs Free Tier is disabled due to unusual activity detected.\n\n"
                                "Possible reasons:\n"
                                "• Using a proxy/VPN\n"
                                "• Multiple free accounts detected\n"
                                "• Account flagged for abuse\n\n"
                                "Solutions:\n"
                                "• Purchase a Paid Plan to continue using the service\n"
                                "• Check if you're using a VPN/proxy and disable it\n"
                                "• Contact ElevenLabs support if you believe this is an error\n"
                                "• Verify your API key is valid and not shared"
                            )
                            print(error_msg)
                            return None
                
                # Fallback: check error string
                if "detected_unusual_activity" in error_str or "Free Tier usage disabled" in error_str:
                    error_msg = (
                        "⚠️ ElevenLabs Free Tier is disabled due to unusual activity detected.\n\n"
                        "Possible reasons:\n"
                        "• Using a proxy/VPN\n"
                        "• Multiple free accounts detected\n"
                        "• Account flagged for abuse\n\n"
                        "Solutions:\n"
                        "• Purchase a Paid Plan to continue using the service\n"
                        "• Check if you're using a VPN/proxy and disable it\n"
                        "• Contact ElevenLabs support if you believe this is an error\n"
                        "• Verify your API key is valid and not shared"
                    )
                    print(error_msg)
                    return None
                elif "unauthorized" in error_str.lower() or "invalid" in error_str.lower():
                    error_msg = (
                        "❌ ElevenLabs API authentication failed.\n\n"
                        "Please check:\n"
                        "• Your ELEVEN_LABS_API_KEY is correct\n"
                        "• Your API key has not expired\n"
                        "• Your account has available credits"
                    )
                    print(error_msg)
                    return None
            
            # Generic error handling
            print(f"Error generating voice response: {e}")
            # Print available methods for debugging (only in verbose mode)
            if "DEBUG" in os.environ:
                print(f"Available client attributes: {[attr for attr in dir(self.client) if not attr.startswith('_')]}")
                if hasattr(self.client, 'text_to_speech'):
                    print(f"Available text_to_speech attributes: {[attr for attr in dir(self.client.text_to_speech) if not attr.startswith('_')]}")
            return None
        
class VoiceAssistantRAG:
    def __init__(self, elevenlabs_api_key=None, deepgram_api_key=None, tts_provider="elevenlabs"):
        self.whisper_model = whisper.load_model("base")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator(
            elevenlabs_api_key=elevenlabs_api_key,
            deepgram_api_key=deepgram_api_key,
            provider=tts_provider
        )

    def setup_vector_store(self, vector_store):
        """Initialize the vector store and QA chain"""
        self.vector_store = vector_store

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            verbose=True,
        )

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        try:
            # List available audio devices for debugging
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            
            # Use the default input device explicitly
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                device=default_input,
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            
            # Flatten the array if it's 2D
            if len(recording.shape) > 1:
                recording = recording.flatten()
            
            # Calculate audio level metrics for debugging
            max_amplitude = float(recording.max())
            min_amplitude = float(recording.min())
            rms = float((recording ** 2).mean() ** 0.5)  # Root Mean Square
            
            # More lenient check: use RMS threshold instead of exact zero
            # RMS of 0.001 is very quiet, so we use a lower threshold
            silence_threshold = 0.0001  # Very low threshold to detect any audio
            
            if rms < silence_threshold and abs(max_amplitude) < silence_threshold:
                # Provide detailed debugging info
                error_msg = (
                    f"No audio detected. Audio levels - Max: {max_amplitude:.6f}, "
                    f"Min: {min_amplitude:.6f}, RMS: {rms:.6f}. "
                    f"Please check:\n"
                    f"1. Microphone is connected and working\n"
                    f"2. Microphone permissions are granted\n"
                    f"3. Microphone is not muted\n"
                    f"4. You are speaking during recording"
                )
                raise ValueError(error_msg)
            
            return recording
        except sd.PortAudioError as e:
            raise RuntimeError(f"Audio device error: {e}. Please check your microphone connection and permissions.")
        except ValueError:
            # Re-raise ValueError as-is (it has our custom message)
            raise
        except Exception as e:
            raise RuntimeError(f"Error recording audio: {e}")

    def transcribe_audio(self, audio_array):
        """Transcribe audio using Whisper"""
        try:
            # Verify audio array is not empty
            if audio_array is None or len(audio_array) == 0:
                raise ValueError("Audio array is empty. Recording may have failed.")
            
            # Check if audio has actual content (not just silence)
            if audio_array.max() == 0.0 and audio_array.min() == 0.0:
                raise ValueError("No audio detected in recording. Please try recording again.")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                sf.write(temp_audio.name, audio_array, self.sample_rate)
                # Set language to English for better accuracy
                result = self.whisper_model.transcribe(temp_audio.name, language="en")
                os.unlink(temp_audio.name)
            
            transcribed_text = result.get("text", "").strip()
            if not transcribed_text:
                raise ValueError("Transcription returned empty text. Audio may be too quiet or unclear.")
            
            return transcribed_text
        except Exception as e:
            raise RuntimeError(f"Transcription error: {e}")

    def generate_response(self, query):
        """Generate response using RAG system"""
        if self.qa_chain is None:
            return "Error: Vector store not initialized"

        response = self.qa_chain.invoke({"question": query})
        return response["answer"]

    def text_to_speech(self, text: str, voice_name: str = None, provider: str = None) -> str:
        """Convert text to speech"""
        return self.voice_generator.generate_voice_response(text, voice_name, provider)
    
    
def setup_knowledge_base():
    st.title("🌙 Lunar - Knowledge Base Setup")

    doc_processor = DocumentProcessor()

    uploaded_files = st.file_uploader(
        "Upload your documents", accept_multiple_files=True, type=["pdf", "txt", "md"]
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            temp_dir = tempfile.mkdtemp()

            # Save uploaded files
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            try:
                # Process documents
                documents = doc_processor.load_documents(temp_dir)
                processed_docs = doc_processor.process_documents(documents)

                # Create vector store
                vector_store = doc_processor.create_vector_store(
                    processed_docs, "knowledge_base"
                )

                # Store in session state
                st.session_state.vector_store = vector_store

                st.success(f"Processed {len(processed_docs)} document chunks!")

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
            finally:
                # Cleanup
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
                
def main():
    
    
    # Page configuration
    st.set_page_config(
        page_title="Lunar - Voice RAG Assistant",
        page_icon="🌙",
        layout="wide"
    )

    # Check for API keys
    elevenlabs_api_key = os.getenv("ELEVEN_LABS_API_KEY") or env_config.eleven_labs_api_key
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")  or env_config.deepgram_api_key
    openai_api_key = os.getenv("OPENAI_API_KEY") or env_config.openai_api_key

    if not openai_api_key:
        st.error("Please set OPENAI_API_KEY in your environment variables")
        return
    
    if not elevenlabs_api_key and not deepgram_api_key:
        st.warning("⚠️ No TTS API key found. Please set either ELEVEN_LABS_API_KEY or DEEPGRAM_API_KEY in your environment variables")

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Setup Knowledge Base", "Voice Assistant"])

    if page == "Setup Knowledge Base":
        vector_store = setup_knowledge_base()
        if vector_store:
            st.session_state.vector_store = vector_store

    else:  # Voice Assistant page
        if "vector_store" not in st.session_state:
            st.error("Please setup knowledge base first!")
            return

        st.title("🌙 Lunar - Voice Assistant RAG System")
        

        # TTS Provider selection
        available_providers = []
        if elevenlabs_api_key:
            available_providers.append("elevenlabs")
        if deepgram_api_key:
            available_providers.append("deepgram")
        
        if not available_providers:
            st.error("No TTS provider available. Please set at least one API key.")
            return
        
        tts_provider = st.sidebar.selectbox(
            "TTS Provider",
            available_providers,
            index=0,
            help="Select the text-to-speech provider to use"
        )

        # Initialize assistant
        assistant = VoiceAssistantRAG(
            elevenlabs_api_key=elevenlabs_api_key,
            deepgram_api_key=deepgram_api_key,
            tts_provider=tts_provider
        )
        # Initialize the vector store and QA chain
        assistant.setup_vector_store(st.session_state.vector_store)

        # Voice selection
        try:
            if tts_provider == "deepgram":
                available_voices = assistant.voice_generator.deepgram_voices
                selected_voice = st.sidebar.selectbox(
                    "Select Voice (Deepgram)",
                    available_voices,
                    index=0,
                    help="Deepgram Aura voices - high quality neural TTS"
                )
            else:
                available_voices = assistant.voice_generator.available_voices
                if available_voices:
                    selected_voice = st.sidebar.selectbox(
                        "Select Voice (ElevenLabs)",
                        available_voices,
                        index=(
                            available_voices.index("Rachel")
                            if "Rachel" in available_voices
                            else 0
                        ),
                    )
                else:
                    st.warning("No voices available. Using default voice.")
                    selected_voice = "Rachel"
        except Exception as e:
            st.error(f"Error loading voices: {e}")
            selected_voice = "Rachel" if tts_provider == "elevenlabs" else "aura-asteria-en"

        # Recording duration
        duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)
        
        # Show audio device information
        with st.sidebar.expander("Audio Device Info"):
            try:
                devices = sd.query_devices()
                default_input = sd.default.device[0]
                st.write(f"**Default Input Device:** {default_input}")
                st.write(f"**Device Name:** {devices[default_input]['name']}")
                st.write(f"**Sample Rate:** {assistant.sample_rate} Hz")
            except Exception as e:
                st.warning(f"Could not query audio devices: {e}")
        
        # Troubleshooting information
        with st.sidebar.expander("🔧 Common Issues to Check"):
            st.markdown("""
            **Microphone permissions:**
            - macOS: System Settings → Privacy & Security → Microphone
            - Grant microphone access to your terminal/IDE
            
            **Microphone selection:**
            - Check the "Audio Device Info" in the sidebar
            - Ensure the correct device is selected
            
            **Browser permissions (if using Streamlit in browser):**
            - Allow microphone access when prompted
            - Check browser settings if access was denied
            
            **Audio level:**
            - Speak clearly and at a normal volume
            - The code now detects if no audio was captured
            """)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start Recording"):
                try:
                    with st.spinner(f"Recording for {duration} seconds... Please speak now!"):
                        audio_data = assistant.record_audio(duration)
                        st.session_state.audio_data = audio_data
                        st.success("Recording completed!")
                except ValueError as e:
                    st.error(f"⚠️ {e}")
                    st.info("💡 Tips: Make sure your microphone is connected and you've granted microphone permissions to your browser/application.")
                except RuntimeError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        with col2:
            if st.button("Process Recording"):
                if "audio_data" not in st.session_state:
                    st.error("Please record audio first!")
                    return

                # Process recording
                try:
                    with st.spinner("Transcribing..."):
                        query = assistant.transcribe_audio(st.session_state.audio_data)
                        if query:
                            st.write("**You said:**", query)
                        else:
                            st.warning("Transcription returned empty text. Please try recording again.")
                            return
                except ValueError as e:
                    st.error(f"⚠️ {e}")
                    st.info("💡 Try speaking louder or closer to the microphone, and ensure your microphone is working.")
                    return
                except RuntimeError as e:
                    st.error(f"❌ {e}")
                    return
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    return

                with st.spinner("Generating response..."):
                    try:
                        response = assistant.generate_response(query)
                        st.write("Response:", response)
                        st.session_state.last_response = response
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        return

                with st.spinner("Converting to speech..."):
                    try:
                        audio_file = assistant.voice_generator.generate_voice_response(
                            response, selected_voice, provider=tts_provider
                        )
                        if audio_file:
                            st.audio(audio_file)
                            os.unlink(audio_file)
                        else:
                            st.warning("⚠️ Voice generation failed. This might be due to:")
                            provider_msg = "ElevenLabs" if tts_provider == "elevenlabs" else "Deepgram"
                            st.info(f"""
                            • **{provider_msg} API issue**: Check if your API key is valid and your account has credits
                            • **Free Tier restrictions**: If using a free account, it may be disabled due to unusual activity
                            • **Network issues**: Check your internet connection
                            
                            Check the console/terminal for detailed error messages.
                            """)
                    except Exception as e:
                        error_msg = str(e)
                        provider_name = "ElevenLabs" if tts_provider == "elevenlabs" else "Deepgram"
                        if "401" in error_msg or "detected_unusual_activity" in error_msg:
                            if tts_provider == "elevenlabs":
                                st.error("⚠️ ElevenLabs API Error: Free Tier may be disabled")
                                st.info("""
                                **Possible solutions:**
                                - Disable VPN/proxy if you're using one
                                - Purchase a Paid Plan from ElevenLabs
                                - Check your API key and account status
                                - Contact ElevenLabs support if you believe this is an error
                                - Try switching to Deepgram TTS provider
                                """)
                            else:
                                st.error(f"⚠️ {provider_name} API Error: Authentication failed")
                                st.info("""
                                **Possible solutions:**
                                - Check your DEEPGRAM_API_KEY is correct
                                - Verify your account has available credits
                                - Check your API key permissions
                                - Try switching to ElevenLabs TTS provider
                                """)
                        else:
                            st.error(f"Error generating voice ({provider_name}): {error_msg}")

        # Display chat history
        if "chat_history" in st.session_state:
            st.subheader("Chat History")
            for q, a in st.session_state.chat_history:
                st.write("Q:", q)
                st.write("A:", a)
                st.write("---")


if __name__ == "__main__":
    # Streamlit apps should be run with: streamlit run voice_assis.py
    # This check prevents running directly with python
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            print("\n" + "="*60)
            print("⚠️  This is a Streamlit application!")
            print("="*60)
            print("\nPlease run it with:")
            print("  streamlit run voice_assis.py\n")
            print("="*60 + "\n")
            import sys
            sys.exit(0)
    except ImportError:
        # If streamlit isn't installed, show error
        print("\nStreamlit is required. Install with: pip install streamlit\n")
        import sys
        sys.exit(1)
    except:
        # If we get here, streamlit context might not be available
        # but let's try to run anyway (streamlit will handle it)
        pass
    
    main()