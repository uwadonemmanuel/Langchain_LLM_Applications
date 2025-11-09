import os
import yt_dlp
import whisper
import streamlit as st
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
import shutil

load_dotenv()

# Set up ffmpeg location for yt_dlp and Whisper
conda_bin = "/usr/local/Caskroom/miniconda/base/bin"
if os.path.exists(conda_bin) and conda_bin not in os.environ.get("PATH", ""):
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{conda_bin}:{current_path}"

# Embedding and LLM models setup
class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.embedding_fn = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif model_type == "chroma":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embedding_fn = HuggingFaceEmbeddings()
        elif model_type == "nomic":
            from langchain_ollama import OllamaEmbeddings
            self.embedding_fn = OllamaEmbeddings(
                model="nomic-embed-text", base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")

class LLMModel:
    def __init__(self, model_type="openai", model_name="gpt-4"):
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is required for OpenAI models")
            self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        elif model_type == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                format="json",
                timeout=120,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")

class YoutubeVideoSummarizer:
    def __init__(self, llm_type="openai", llm_model_name="gpt-4", embedding_type="openai"):
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)
        self.whisper_model = whisper.load_model("base")

    def get_model_info(self) -> Dict:
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def download_video(self, url: str) -> tuple:
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
            "outtmpl": "downloads/%(title)s.%(ext)s",
        }
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            ydl_opts["ffmpeg_location"] = os.path.dirname(ffmpeg_path)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
            video_title = info.get("title", "Unknown Title")
            return audio_path, video_title

    def transcribe_audio(self, audio_path: str) -> str:
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def create_documents(self, text: str, video_title: str) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
        texts = text_splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"source": video_title}) for chunk in texts]

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        return Chroma.from_documents(documents=documents, embedding=self.embedding_model.embedding_fn, collection_name=f"youtube_summary_{self.embedding_model.model_type}")

    def generate_summary(self, documents: List[Document]) -> str:
        map_prompt = ChatPromptTemplate.from_template("""Write a concise summary of the following transcript section: "{text}" CONCISE SUMMARY:""")
        combine_prompt = ChatPromptTemplate.from_template("""Write a detailed summary of the following video transcript sections: "{text}" DETAILED SUMMARY:""")
        summary_chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )
        return summary_chain.invoke(documents)

    def setup_qa_chain(self, vector_store: Chroma):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm_model.llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True,
        )

    def process_video(self, url: str) -> Dict:
        try:
            os.makedirs("downloads", exist_ok=True)
            audio_path, video_title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, video_title)
            summary = self.generate_summary(documents)
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)

            os.remove(audio_path)
            return {
                "summary": summary,
                "qa_chain": qa_chain,
                "title": video_title,
                "full_transcript": transcript,
            }
        except Exception as e:
            return {"error": str(e)}

# Streamlit app
def main():
    # Page configuration
    st.set_page_config(
        page_title="Lunar - YouTube Video Summarizer",
        page_icon="🌙",
        layout="wide"
    )
    
    st.title("🌙 Lunar - YouTube Video Summarizer")
    st.sidebar.header("Settings")
    
    # Select model options
    llm_choice = st.sidebar.selectbox("Choose LLM model", ["OpenAI GPT-4", "Ollama Llama3.2"])
    embedding_choice = st.sidebar.selectbox("Choose Embedding", ["OpenAI", "Chroma Default", "Nomic (via Ollama)"])

    # URL input
    url = st.text_input("Enter YouTube URL:")
    if url:
        with st.spinner("Processing video..."):
            summarizer = YoutubeVideoSummarizer(
                llm_type="openai" if llm_choice == "OpenAI GPT-4" else "ollama",
                llm_model_name="gpt-4" if llm_choice == "OpenAI GPT-4" else "llama3.2",
                embedding_type="openai" if embedding_choice == "OpenAI" else "chroma" if embedding_choice == "Chroma Default" else "nomic"
            )

            result = summarizer.process_video(url)

            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                st.subheader(f"Video Title: {result['title']}")
                st.subheader("Summary:")
                st.write(result["summary"])

                # Interactive Q&A
                query = st.text_input("Ask a question about the video:")
                if query:
                    response = result["qa_chain"].invoke({"question": query})
                    st.write(f"Answer: {response['answer']}")

                # Full transcript option
                if st.button("Show full transcript"):
                    st.subheader("Full Transcript:")
                    st.write(result["full_transcript"])

if __name__ == "__main__":
    main()
