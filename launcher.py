import streamlit as st
import subprocess
import sys
import webbrowser
import time
import socket
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Streamlit LLM Apps Launcher",
    page_icon="🚀",
    layout="wide"
)

# Get the current directory
current_dir = Path(__file__).parent

# Define available apps (ports will be assigned dynamically)
AVAILABLE_APPS = {
    "Voice Assistant RAG": {
        "file": "voice_assist_streamlit.py",
        "description": "Lunar - Voice-powered RAG assistant with document processing and TTS",
        "icon": "🎤"
    },
    "News Summarizer": {
        "file": "news_summarizer_streamlit.py",
        "description": "Lunar - Your friendly News Summarizer",
        "icon": "🌙"
    },
    "YouTube Summarizer": {
        "file": "yt_vid_summarizer_streamlit.py",
        "description": "Lunar - Your friendly YouTube Video Summarizer",
        "icon": "📺"
    }
}

# -------------------------------
# Utility: find an available port
# -------------------------------
def get_free_port(start_port=8501, end_port=8600):
    """Find an available TCP port within a range."""
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
            except OSError:
                continue
    raise RuntimeError("⚠️ No free ports available between 8501–8600.")

# -------------------------------
# Function to launch an app
# -------------------------------
def launch_app(app_file: str, port: int = None):
    """Launch a Streamlit app in a subprocess on a specific port."""
    app_path = current_dir / app_file
    if not app_path.exists():
        st.error(f"❌ App file not found: {app_file}")
        return False

    # Assign a dynamic port if not provided
    if port is None:
        port = get_free_port()

    try:
        # Start Streamlit app as a subprocess
        process = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                str(app_path),
                "--server.port", str(port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Give the server time to start
        time.sleep(2)

        # Open browser
        url = f"http://localhost:{port}"
        try:
            webbrowser.open(url)
        except:
            pass

        return port
    except Exception as e:
        st.error(f"❌ Error launching app: {e}")
        return False

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.title("🚀 Langchain - LLM Apps Launcher")
    st.markdown("Select an app to launch it in a new browser tab.")
    
    if "launched_apps" not in st.session_state:
        st.session_state.launched_apps = {}

    # Sidebar
    with st.sidebar:
        st.header("📱 Available Apps")
        selected_app = st.selectbox(
            "Select App to Launch",
            list(AVAILABLE_APPS.keys()),
            key="app_selector"
        )

        if selected_app:
            app_info = AVAILABLE_APPS[selected_app]
            app_key = app_info['file']
            is_running = app_key in st.session_state.launched_apps

            st.markdown(f"### {app_info['icon']} {selected_app}")
            st.markdown(f"**File:** `{app_info['file']}`")
            st.markdown(f"**Description:** {app_info['description']}")
            st.markdown("---")

            if is_running:
                port = st.session_state.launched_apps[app_key]['port']
                st.success(f"✅ Running at http://localhost:{port}")

                if st.button("🔄 Relaunch App", use_container_width=True):
                    new_port = launch_app(app_info['file'])
                    if new_port:
                        st.session_state.launched_apps[app_key] = {
                            'port': new_port,
                            'file': app_info['file']
                        }
                        st.rerun()
            else:
                if st.button("🚀 Launch App", use_container_width=True, type="primary"):
                    new_port = launch_app(app_info['file'])
                    if new_port:
                        st.session_state.launched_apps[app_key] = {
                            'port': new_port,
                            'file': app_info['file']
                        }
                        st.success(f"✅ Launched {selected_app} on port {new_port}")
                        st.info(f"🌐 Opening http://localhost:{new_port}")
                        st.rerun()

    # Main content
    st.markdown("---")
    st.subheader("🟢 Running Apps")

    if not st.session_state.launched_apps:
        st.info("No apps are currently running.")
    else:
        for app_file, app_data in st.session_state.launched_apps.items():
            app_name = next((n for n, info in AVAILABLE_APPS.items() if info["file"] == app_file), app_file)
            port = app_data["port"]
            st.markdown(f"**{app_name}** ({app_file}) — [http://localhost:{port}](http://localhost:{port})")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:gray'><small>Dynamic Streamlit Launcher — Automatically assigns free ports</small></div>",
        unsafe_allow_html=True
    )

# -------------------------------
# Run main
# -------------------------------
if __name__ == "__main__":
    main()
