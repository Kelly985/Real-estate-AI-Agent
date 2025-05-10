import streamlit as st
import sys
import traceback
import numpy as np
import sounddevice as sd
from datetime import datetime
import logging
from ai_agent import AudioHandler, AIAgent
import pkg_resources
import os

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
        'is_recording': False,
        'conversation': [],
        'status': "Idle",
        'debug_mode': False,
        'initialized': False,
        'session_initialized': False,
        'llm_logs': []
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
        st.write("Ensure all dependencies are installed, microphone permissions are granted, and TOGETHER_API_KEY is set.")
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
        st.error(f"Streamlit version {streamlit_version} is outdated. Please upgrade to 1.29.0: `pip install streamlit==1.29.0`")
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
            deps = ['streamlit', 'torch', 'faster_whisper', 'sentence_transformers', 'together', 'pillow', 'backoff', 'sounddevice', 'numpy', 'pyttsx3']
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
            import sounddevice as sd
            from ai_agent import AudioHandler, AIAgent
            devices = sd.query_devices()
            logger.info(f"Available audio devices at startup: {devices}")
        except ImportError as e:
            st.error("Missing critical dependencies. Run `pip install -r requirements.txt`.")
            debug_mode = st.session_state.get('debug_mode', False)
            if debug_mode:
                st.code(traceback.format_exc(), language='python')
            return
        except Exception as e:
            st.error("Failed to initialize audio devices. Ensure a microphone is connected.")
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
            response_suffix = " (TTS Played)" if entry.get("tts_played", False) else " (TTS Failed)" if entry["mode"] == "voice" else ""
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
        
        if st.button("Test Microphone", key="test_mic"):
            try:
                devices = sd.query_devices()
                logger.info(f"Test microphone - Available audio devices: {devices}")
                audio_data = audio.record_audio(duration=1.0)
                st.success(f"Microphone detected! Recorded {len(audio_data)} samples.")
            except Exception as e:
                st.error(f"Microphone test failed: {str(e)}")
                logger.error(f"Microphone test error: {str(e)}")
                if st.session_state.get('debug_mode', False):
                    st.code(traceback.format_exc(), language='python')
        
        try:
            devices = sd.query_devices()
            logger.info(f"Voice mode - Available audio devices: {devices}")
        except Exception as e:
            st.error("Microphone not detected. Please ensure a microphone is connected and permissions are granted.")
            st.session_state.status = "Microphone Error"
            status_container.write(f"**Status**: {st.session_state.status}")
            logger.error(f"Microphone detection error: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.code(traceback.format_exc(), language='python')
            return
        
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
            logger.warning("Manually initialized is_recording")
        
        if st.button("Speak", key="speak_button", help="Click to record for 6 seconds"):
            if not st.session_state.is_recording:
                st.session_state.is_recording = True
                st.session_state.status = "Recording..."
                status_container.write(f"**Status**: {st.session_state.status}")
                record_and_process(agent, audio, status_container)
        
        if st.button("Clear Session State", key="clear_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.session_initialized = False
            st.session_state.llm_logs = []
            st.rerun()
        
        if st.session_state.get('debug_mode', False):
            st.write("#### Debug Logs")
            if st.session_state.llm_logs:
                st.write("**Recent LLM Logs**:")
                for log in st.session_state.llm_logs[-5:]:
                    st.write(log)
            if st.session_state.is_recording and st.session_state.status not in ["Recording...","Processing...","Responding..."]:
                st.warning("Recording state is stuck. Try refreshing or clicking 'Clear Session State'.")
    
    else:
        st.write("#### Text Interaction")
        # Initialize query_input in session state if not present
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

def record_and_process(agent, audio, status_container):
    """Record audio for 6 seconds and process in main thread."""
    logger.debug("Starting record and process")
    
    try:
        devices = sd.query_devices()
        logger.info(f"Recording - Available audio devices: {devices}")
        st.session_state.status = "Recording..."
        status_container.write(f"**Status**: {st.session_state.status}")
        audio_data = audio.record_audio(duration=6.0)
        
        st.session_state.status = "Processing..."
        status_container.write(f"**Status**: {st.session_state.status}")
        query = audio.transcribe(audio_data)
        logger.debug(f"Transcribed query: {query}")
        st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Transcribed query: {query}")
        
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
                logger.info(f"Executing TTS in main thread: {response[:50]}...")
                tts_success = audio.text_to_speech(response)
                if tts_success:
                    st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] TTS played: {response[:50]}...")
                else:
                    st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Error: TTS failed - Execution unsuccessful")
                    st.error("TTS failed. Ensure pyttsx3 is installed correctly.")
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
            # Force rerun to update chat history
            st.rerun()
        
        st.session_state.status = "Idle"
        status_container.write(f"**Status**: {st.session_state.status}")
        logger.debug("Transitioned to Idle state")
    
    except sd.PortAudioError as e:
        logger.error(f"Microphone error: {str(e)}")
        st.session_state.status = "Microphone Error"
        st.error("Microphone error: Ensure microphone is connected and permissions are granted")
        st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Microphone error: {str(e)}")
    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        st.session_state.status = "Error"
        st.error(f"Error: Recording failed - {str(e)}")
        st.session_state.llm_logs.append(f"[{datetime.now().isoformat()}] Error: Recording failed - {str(e)}")
    finally:
        st.session_state.is_recording = False
        logger.debug("Record and process completed")

if __name__ == "__main__":
    main()