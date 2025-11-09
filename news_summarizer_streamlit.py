import os
import streamlit as st
from typing import Optional
import requests
from appconfig import env_config 
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from newspaper import Article
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

openai_api_key = os.getenv("OPENAI_API_KEY") or env_config.openai_api_key  # Retrieve the API key from the config

class NewsArticleSummarizer:
    def __init__(
        self,
        api_key: str = openai_api_key,  # Default API key from appconfig if none provided
        model_provider: str = "openai",
        model_name: str = "gpt-4",  # Default model for OpenAI
    ):
        """
        Initialize the summarizer with choice of model provider and model.
        Args:
            api_key: OpenAI API key (required for OpenAI models)
            model_provider: 'openai' or 'ollama'
            model_name: specific model name (e.g., 'gpt-4' for OpenAI or 'llama3.2' for Ollama)
        """
        self.model_provider = model_provider
        self.model_name = model_name

        # Setup LLM based on model provider and model name
        if model_provider == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI models")
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        elif model_provider == "ollama":
            # Check if Ollama is running before initializing
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                response.raise_for_status()
            except Exception as e:
                raise ConnectionError(
                    f"Ollama is not running or not accessible at localhost:11434.\n"
                    f"Please start Ollama by running: ollama serve\n"
                    f"Original error: {e}"
                )
            
            # Using ChatOllama with proper configuration
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                format="json",  # Optional: for structured output
                timeout=120,  # Increased timeout for longer generations
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

        # Initialize text splitter for long articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=len
        )

    def fetch_article(self, url: str) -> Optional[Article]:
        """Fetch article content using newspaper3k"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            st.error(f"Error fetching article: {e}")
            return None

    def create_documents(self, text: str) -> list[Document]:
        """Create LangChain documents from text"""
        texts = self.text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        return docs

    def summarize(self, url: str, summary_type: str = "detailed") -> dict:
        """Main summarization pipeline"""
        # Fetch article
        article = self.fetch_article(url)
        if not article:
            return {"error": "Failed to fetch article"}

        # Create documents
        docs = self.create_documents(article.text)

        # Define prompts based on summary type
        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of the following text:
            "{text}"
            DETAILED SUMMARY:"""

            combine_prompt_template = """Write a detailed summary of the following text that combines the previous summaries:
            "{text}"
            FINAL DETAILED SUMMARY:"""
        else:  # concise summary
            map_prompt_template = """Write a concise summary of the following text:
            "{text}"
            CONCISE SUMMARY:"""

            combine_prompt_template = """Write a concise summary of the following text that combines the previous summaries:
            "{text}"
            FINAL CONCISE SUMMARY:"""

        # Create prompts
        map_prompt = PromptTemplate(
            template=map_prompt_template, input_variables=["text"]
        )

        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        # Create and run chain
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )

        # Generate summary with error handling
        try:
            summary = chain.invoke(docs)
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "Errno 61" in error_msg:
                return {
                    "error": "Ollama server is not running. Please start it with: ollama serve",
                    "title": article.title if article else "Unknown",
                    "url": url,
                }
            else:
                return {
                    "error": f"Error generating summary: {error_msg}",
                    "title": article.title if article else "Unknown",
                    "url": url,
                }

        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "summary": summary,
            "url": url,
            "model_info": {"provider": self.model_provider, "model_name": self.model_name},
        }

# Streamlit UI for the app
def streamlit_app():
    st.title("🌙 Lunar - News Article Summarizer")
    st.set_page_config(page_title="Lunar - News Article Summarizer", page_icon="🌙", layout="centered")
    

    # Input URL
    url = st.text_input("Enter the URL of the news article:")

    # Select summary type
    summary_type = st.selectbox("Select summary type", ["detailed", "concise"])

    # Select model provider (OpenAI or Ollama)
    model_provider = st.selectbox("Select model provider", ["openai", "ollama"])

    # Select model based on the selected provider
    if model_provider == "openai":
        model_name = st.selectbox("Select OpenAI model", ["gpt-5", "gpt-4", "gpt-3.5-turbo"])
    elif model_provider == "ollama":
        model_name = st.selectbox("Select Ollama model", ["llama3.2", "nomic-embed-text"])

    # If OpenAI model is selected, input API key
    if model_provider == "openai":
        api_key = st.text_input("Enter OpenAI API Key (leave empty to use default)", type="password")
        
        # Use default API key from appconfig if the user didn't provide one
        if not api_key:
            api_key = openai_api_key  # Default API key from appconfig
            if api_key:
                st.info("Using default API key from environment variables.")
            else:
                st.warning("No API key provided, and no default API key found. Please provide an API key.")
    else:
        api_key = None

    # When user submits the URL
    if st.button("Generate Summary"):
        if not url:
            st.error("Please enter a valid URL.")
            return
        
        if model_provider == "openai" and not api_key:
            st.error("Please provide your OpenAI API key.")
            return

        # Initialize the summarizer with the selected model provider and model
        summarizer = NewsArticleSummarizer(api_key=api_key, model_provider=model_provider, model_name=model_name)

        with st.spinner("Fetching and summarizing article..."):
            summary_result = summarizer.summarize(url, summary_type)

        
                        
        # Show summary results
        if "error" in summary_result:
            st.error(f"❌ Error: {summary_result['error']}")
        else:
            # Display Title and Meta Information
            st.subheader(f"Title: {summary_result['title']}")
            st.write(f"Model: {summary_result.get('model_info', {}).get('provider', 'Unknown')} - {summary_result.get('model_info', {}).get('model_name', 'Unknown')}")

            # If model name is 'ollama - llama3.2' or similar, display custom message
            if "ollama" in summary_result.get('model_info', {}).get('provider', '').lower():
                st.markdown("💡 *Using Ollama’s LLaMA 3.2 model for summarization.*")

            # Display Summary Text
            summary_data = summary_result.get('summary', {})

            if isinstance(summary_data, dict):
                summary_text = summary_data.get("output_text", "") or summary_data.get("text", "")
            else:
                summary_text = str(summary_data)

            # Present the summary in a scrollable text box
            st.text_area(
                "Summary:",
                summary_text.strip(),
                height=350,
                disabled=False  # Makes it read-only
            )

            # Display Source Documents if available
            input_docs = summary_result.get("input_documents", [])
            if input_docs:
                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(input_docs, 1):
                        st.markdown(f"### Document {i}")

                        # Extract page content based on document type
                        if isinstance(doc, dict):
                            content = doc.get("page_content", "")
                        elif isinstance(doc, str) and "page_content=" in doc:
                            # Extract text from stringified `Document()`
                            import re
                            match = re.search(r"page_content='(.*?)'\)", doc, re.DOTALL)
                            content = match.group(1).strip() if match else doc
                        else:
                            content = str(doc)

                        # Truncate long content for readability
                        display_text = content[:1000] + "..." if len(content) > 1000 else content

                        st.text_area(
                            f"Content (Document {i})",
                            display_text,
                            height=300,
                            disabled=False  # Makes it read-only
                        )

# Run the app
if __name__ == "__main__":
    streamlit_app()
