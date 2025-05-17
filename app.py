import os
os.environ['STREAMLIT_SERVER_PORT'] = os.getenv('PORT', '10000')
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

import streamlit as st
import sys
import traceback
import numpy as np
from datetime import datetime
import logging
from ai_agent import AudioHandler, AIAgent
import pkg_resources
import pydub
import hashlib


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000

# CSS for chat interface, history, and input styling
CHAT_CSS = """
<style>
.chat-container {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 10px;
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 120px; /* Space for input box */
}
.customer-message {
    background-color: #E6F3FF;
    padding: 10px;
    margin: 5px;
    border-radius: 10px;
    max-width: 70%;
    float: left;
    clear: both;
}
.agent-message {
    background-color: #E6FFE6;
    padding: 10px;
    margin: 5px;
    border-radius: 10px;
    max-width: 70%;
    float: right;
    clear: both;
}
.speak-button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 16px;
    cursor: pointer;
    user-select: none;
}
.speak-button:hover {
    background-color: #45a049;
}
.input-container {
    padding: 10px;
    background-color: #ffffff;
    border-top: 1px solid #e0e0e0;
}
</style>
"""

def initialize_session_state():
    """Initialize session state with default values."""
    if 'session_initialized' in st.session_state and st.session_state.session_initialized:
        logger.debug("Session state already initialized")
        return
    
    default_state = {
        'agent': None,
        'audio': None,
        'is_processing': False,
        'conversation': [],
        'status': "Idle",
        'debug_mode': False,
        'initialized': False,
        'session_initialized': False,
        'llm_logs': [],
        'last_audio_hash': None,  # Track hash of last processed audio
        'uploader_key': 0,  # Dynamic key for file_uploader
        'processed_files': set(),  # Track processed file hashes
        'pending_audio': None  # Store audio file temporarily
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
            logger.debug(f"Initialized session state key: {key}")
    st.session_state.session_initialized = True
    logger.info("Session state initialization complete")

def initialize_components():
    """Initialize AIAgent and AudioHandler."""
    if getattr(st.session_state, 'initialized', False):
        return True
    
    initialize_session_state()
    
    st.write("### Initialization Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Initialize AIAgent
        status_text.text("Initializing AIAgent... (1/2)")
        progress_bar.progress(50)
        
        try:
            st.session_state.agent = AIAgent()
        except Exception as e:
            if "selected index k out of range" in str(e):
                logger.warning("Knowledge base empty - initializing with empty state")
                st.session_state.agent = AIAgent()
                st.session_state.agent.knowledge_base.knowledge = {
                    "questions": [], "answers": [], "sources": [], "embeddings": None
                }
            else:
                raise
        
        # Step 2: Initialize AudioHandler
        status_text.text("Initializing AudioHandler... (2/2)")
        progress_bar.progress(75)
        st.session_state.audio = AudioHandler()
        
        progress_bar.progress(100)
        status_text.text("✅ All components loaded successfully!")
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error("### CRITICAL INITIALIZATION ERROR")
        st.write("Ensure all dependencies are installed and TOGETHER_API_KEY is set.")
        debug_mode = st.session_state.get('debug_mode', False)
        if debug_mode:
            st.code(traceback.format_exc(), language='python')
        st.session_state.agent = None
        st.session_state.audio = None
        st.session_state.initialized = False
        return False

def main():
    # Ensure session state is initialized
    try:
        initialize_session_state()
    except Exception as e:
        st.error(f"Failed to initialize session state: {str(e)}")
        logger.error(f"Session state initialization in main failed: {str(e)}")
        return
    
    st.set_page_config(page_title="Real Estate Voice Agent", layout="wide")
    
    import streamlit
    streamlit_version = streamlit.__version__
    if streamlit_version < "1.10.0":
        st.error(f"Streamlit version {streamlit_version} is outdated. Please upgrade to 1.45.0: `pip install streamlit==1.45.0`")
        return
    
    try:
        st.title("🏡 Real Estate Voice Agent")
        
        st.markdown(CHAT_CSS, unsafe_allow_html=True)
        
        # Initialize debug_mode if not set
        if 'debug_mode' not in st.session_state:
            st.session_state['debug_mode'] = False
        st.session_state['debug_mode'] = st.checkbox("Enable Debug Mode", value=st.session_state.get('debug_mode', False))
        logger.debug(f"Debug mode set to: {st.session_state.get('debug_mode', False)}")
        
        if st.session_state.get('debug_mode', False):
            st.write("#### Dependency Versions")
            deps = ['streamlit', 'torch', 'faster_whisper', 'sentence_transformers', 'together', 'pillow', 'backoff', 'numpy', 'gTTS', 'pydub']
            versions = {}
            for dep in deps:
                try:
                    version = pkg_resources.get_distribution(dep).version
                    versions[dep] = version
                except:
                    versions[dep] = "Not installed"
            st.write(versions)
        
        try:
            import numpy as np
            import pydub
            from ai_agent import AudioHandler, AIAgent
        except ImportError as e:
            st.error("Missing critical dependencies. Run `pip install -r requirements.txt`.")
            debug_mode = st.session_state.get('debug_mode', False)
            if debug_mode:
                st.code(traceback.format_exc(), language='python')
            return
        
        if not initialize_components():
            return
        
        agent = getattr(st.session_state, 'agent', None)
        audio = getattr(st.session_state, 'audio', None)
        if agent is None or audio is None:
            st.error("Failed to initialize AI components. Please refresh and try again.")
            debug_mode = st.session_state.get('debug_mode', False)
            if debug_mode:
                st.code("Agent or AudioHandler is None", language='python')
            return
        show_main_interface(agent, audio)
    
    except Exception as e:
        st.error("### APPLICATION ERROR")
        st.write("An unexpected error occurred. Please refresh and try again.")
        debug_mode = st.session_state.get('debug_mode', False)
        if debug_mode:
            st.code(traceback.format_exc(), language='python')
        logger.error(f"Main function error: {str(e)}")

def show_main_interface(agent, audio):
    """The main interactive interface."""
    st.write("### Conversation Interface")
    
    input_mode = st.radio("Input Mode", ["Voice", "Text"])
    
    # Chat History Section
    st.write("#### Chat History")
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for entry in st.session_state.conversation:
            query_prefix = "🎙️ Voice Query: " if entry["mode"] == "voice" else "Text Query: "
            st.markdown(
                f'<div class="customer-message"><b>{entry["timestamp"]} - You</b><br>{query_prefix}{entry["query"]}</div>',
                unsafe_allow_html=True
            )
            response_suffix = " (Audio Played)" if entry.get("tts_played", False) else " (Audio Failed)" if entry["mode"] == "voice" else ""
            st.markdown(
                f'<div class="agent-message"><b>{entry["timestamp"]} - Comfy AI</b><br>{entry["response"]}{response_suffix}</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input Section (Voice or Text) at the Bottom
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    if input_mode == "Voice":
        st.write("#### Voice Interaction")
        status_container = st.empty()
        status_container.write(f"**Status**: {st.session_state.status}")
        
        # Use dynamic key to reset file_uploader
        uploader_key = f"audio_upload_{st.session_state.uploader_key}"
        audio_file = st.file_uploader(
            "Upload an audio file (WAV or MP3)",
            type=["wav", "mp3"],
            key=uploader_key
        )
        
        if audio_file and not st.session_state.is_processing:
            # Compute file hash to identify unique files
            audio_data = audio_file.read()
            audio_hash = hashlib.md5(audio_data).hexdigest()
            audio_file.seek(0)  # Reset file pointer
            logger.debug(f"Computed audio hash: {audio_hash}, Last hash: {st.session_state.last_audio_hash}")
            
            if audio_hash not in st.session_state.processed_files and audio_hash != st.session_state.last_audio_hash:
                st.session_state.is_processing = True
                st.session_state.status = "Processing..."
                status_container.write(f"**Status**: {st.session_state.status}")
                st.session_state.last_audio_hash = audio_hash
                st.session_state.processed_files.add(audio_hash)
                st.session_state.pending_audio = audio_file  # Store file temporarily
                # Increment uploader_key to reset uploader
                st.session_state.uploader_key += 1
                logger.debug(f"Incremented uploader_key to: audio_upload_{st.session_state.uploader_key}")
                
                # Process the audio file
                record_and_process(agent, audio, status_container, st.session_state.pending_audio, audio_hash)
            else:
                logger.debug(f"Skipping already processed audio file with hash: {audio_hash}")
                # Clear uploader by incrementing key
                st.session_state.uploader_key += 1
                logger.debug(f"Incremented uploader_key for skipped file to: audio_upload_{st.session_state.uploader_key}")
        
        if st.button("Clear Session State", key="clear_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.session_initialized = False
            st.session_state.llm_logs = []
            st.session_state.processed_files = set()  # Reset processed files
            st.rerun()
        
        if st.session_state.get('debug_mode', False):
            st.write("#### Debug Logs")
            if st.session_state.llm_logs:
                st.write("**Recent LLM Logs**:")
                for log in st.session_state.llm_logs[-5:]:
                    st.write(log)
            if st.session_state.is_processing and st.session_state.status not in ["Processing...","Responding..."]:
                st.warning("Processing state is stuck. Try refreshing or clicking 'Clear Session State'.")
    
    else:
        st.write("#### Text Interaction")
        # Initialize query_input in session state if not set
        if 'query_input' not in st.session_state:
            st.session_state.query_input = ""
        
        with st.form(key="text_query_form", clear_on_submit=True):
            query = st.text_input("Enter your question:", value=st.session_state.query_input, key="text_query_input")
            submit_button = st.form_submit_button("Submit")
            
            if submit_button and query:
                with st.spinner("Generating response..."):
                    try:
                        response = agent.generate_response(query, mode="text")
                        st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Query: {query}, Response: {response[:50]}...")
                    except Exception as e:
                        response = f"Error generating response: {str(e)}"
                        st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Error: LLM failed - {str(e)}")
                        if "rate limit" in str(e).lower():
                            st.warning("LLM rate limit reached. Please wait a minute or contact Together AI for higher limits.")
                    timestamp = datetime.now().strftime("%Y-%m-d %H:%M:%S")
                    st.session_state.conversation.append({
                        "timestamp": timestamp,
                        "query": query,
                        "response": response,
                        "mode": "text"
                    })
                    # Clear the text input
                    st.session_state.query_input = ""
                    # Force rerun to update chat history
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def record_and_process(agent, audio, status_container, audio_file, audio_hash):
    """Process uploaded audio file and generate response."""
    logger.debug(f"Starting audio file processing with hash: {audio_hash}")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_file.read())
        
        # Convert to WAV with pydub for faster_whisper
        sound = pydub.AudioSegment.from_file(temp_path)
        sound = sound.set_channels(1).set_frame_rate(SAMPLE_RATE)
        sound.export(temp_path, format="wav")
        
        st.session_state.status = "Processing..."
        status_container.write(f"**Status**: {st.session_state.status}")
        query = audio.transcribe(temp_path)
        logger.debug(f"Transcribed query: {query}")
        st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Transcribed query: {query}")
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if query.strip():
            try:
                response = agent.generate_response(query, mode="voice")
                st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Query: {query}, Response: {response[:50]}...")
            except Exception as e:
                response = f"Error generating response: {str(e)}"
                st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Error: LLM failed - {str(e)}")
                if "rate limit" in str(e).lower():
                    st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Warning: LLM rate limit reached")
                st.error(f"LLM failed: {str(e)}")
            
            timestamp = datetime.now().strftime("%Y-%m-d %H:%M:%S")
            tts_success = False
            st.session_state.status = "Responding..."
            status_container.write(f"**Status**: {st.session_state.status}")
            try:
                logger.info(f"Generating TTS: {response[:50]}...")
                audio_file_path = audio.text_to_speech(response)
                if audio_file_path:
                    st.audio(audio_file_path)
                    tts_success = True
                    st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] TTS played: {response[:50]}...")
                else:
                    st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Error: TTS failed - Generation unsuccessful")
                    st.error("TTS failed. Ensure gTTS is installed correctly.")
            except Exception as e:
                st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Error: TTS failed - {str(e)}")
                st.error(f"TTS failed: {str(e)}")
            
            st.session_state.conversation.append({
                "timestamp": timestamp,
                "query": query,
                "response": response,
                "mode": "voice",
                "tts_played": tts_success
            })
            
            st.session_state.status = "Idle"
            status_container.write(f"**Status**: {st.session_state.status}")
            logger.debug("Transitioned to Idle state")
            
            # Clear pending audio and force rerun
            st.session_state.pending_audio = None
            st.rerun()
        
        else:
            st.session_state.status = "Idle"
            status_container.write(f"**Status**: {st.session_state.status}")
            logger.debug("No valid transcription; transitioned to Idle state")
    
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        st.session_state.status = "Error"
        st.error(f"Error: Audio processing failed - {str(e)}")
        st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Error: Audio processing failed - {str(e)}")
    finally:
        st.session_state.is_processing = False
        st.session_state.pending_audio = None
        # Increment uploader_key to ensure uploader is reset
        st.session_state.uploader_key += 1
        logger.debug(f"Final increment of uploader_key to: audio_upload_{st.session_state.uploader_key}")
        logger.debug("Audio processing completed")

if __name__ == "__main__":
    main()
