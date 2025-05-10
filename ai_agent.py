# import json
# import logging
# import os
# import re
# import time
# import uuid
# from datetime import datetime

# import backoff
# import numpy as np
# import requests
# import sounddevice as sd
# from dotenv import load_dotenv
# from faster_whisper import WhisperModel
# from sentence_transformers import SentenceTransformer
# from together import Together
# import pyttsx3
# import pdfplumber
# import faiss
# from transformers import pipeline

# # Logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Filter to suppress torch.classes warning
# class TorchClassesFilter(logging.Filter):
#     def filter(self, record):
#         return not ("Examining the path of torch.classes raised" in record.getMessage())

# logging.getLogger().addFilter(TorchClassesFilter())

# # Load environment variables
# load_dotenv()

# class KnowledgeBase:
#     """Handles loading, searching, and updating knowledge base with Q&A and documents."""
#     def __init__(self, kb_dir="knowledge_base", chunk_size=300):
#         self.kb_dir = kb_dir
#         self.chunk_size = chunk_size
#         self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
#         self.dimension = self.embedder.get_sentence_embedding_dimension()
#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.knowledge = {
#             "texts": [],
#             "answers": [],
#             "sources": [],
#             "embeddings": []
#         }
#         self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#         self._load_knowledge()

#     def _chunk_text(self, text):
#         """Split text into chunks of approximately chunk_size words."""
#         words = text.split()
#         chunks = []
#         current_chunk = []
#         current_length = 0

#         for word in words:
#             current_chunk.append(word)
#             current_length += 1
#             if current_length >= self.chunk_size:
#                 chunks.append(" ".join(current_chunk))
#                 current_chunk = []
#                 current_length = 0
#         if current_chunk:
#             chunks.append(" ".join(current_chunk))
#         return chunks

#     def _load_knowledge(self):
#         """Load Q&A files and documents from directory."""
#         if not os.path.exists(self.kb_dir):
#             os.makedirs(self.kb_dir)
#             logger.warning(f"Created empty knowledge base directory at {self.kb_dir}")
#             return

#         for filename in os.listdir(self.kb_dir):
#             if filename.endswith(".txt"):
#                 filepath = os.path.join(self.kb_dir, filename)
#                 with open(filepath, 'r', encoding='utf-8') as f:
#                     lines = f.readlines()

#                 current_q = None
#                 for line in lines:
#                     line = line.strip()
#                     if line.startswith("Q:"):
#                         current_q = line[3:].strip()
#                     elif line.startswith("A:") and current_q:
#                         self.knowledge["texts"].append(current_q)
#                         self.knowledge["answers"].append(line[3:].strip())
#                         self.knowledge["sources"].append(filename)
#                         current_q = None

#             elif filename.endswith(".pdf"):
#                 filepath = os.path.join(self.kb_dir, filename)
#                 with pdfplumber.open(filepath) as pdf:
#                     text = " ".join(page.extract_text() or "" for page in pdf.pages)
#                 chunks = self._chunk_text(text)
#                 for chunk in chunks:
#                     self.knowledge["texts"].append(chunk)
#                     self.knowledge["answers"].append(None)
#                     self.knowledge["sources"].append(filename)

#         if self.knowledge["texts"]:
#             logger.info(f"Loaded {len(self.knowledge['texts'])} items (Q&A + document chunks)")
#             self.knowledge["embeddings"] = self.embedder.encode(
#                 self.knowledge["texts"], convert_to_numpy=True
#             )
#             self.index.add(self.knowledge["embeddings"])
#         else:
#             logger.warning("No valid data found in knowledge base")

#     def upload_document(self, file_path):
#         """Process and add a new document to the knowledge base."""
#         try:
#             filename = os.path.basename(file_path)
#             dest_path = os.path.join(self.kb_dir, filename)
#             os.rename(file_path, dest_path)

#             if filename.endswith(".pdf"):
#                 with pdfplumber.open(dest_path) as pdf:
#                     text = " ".join(page.extract_text() or "" for page in pdf.pages)
#                 chunks = self._chunk_text(text)
#             elif filename.endswith(".txt"):
#                 with open(dest_path, 'r', encoding='utf-8') as f:
#                     text = f.read()
#                 chunks = self._chunk_text(text)
#             else:
#                 logger.error(f"Unsupported file type: {filename}")
#                 return False

#             for chunk in chunks:
#                 self.knowledge["texts"].append(chunk)
#                 self.knowledge["answers"].append(None)
#                 self.knowledge["sources"].append(filename)
#                 embedding = self.embedder.encode([chunk], convert_to_numpy=True)
#                 self.knowledge["embeddings"] = np.vstack(
#                     [self.knowledge["embeddings"], embedding]
#                 ) if self.knowledge["embeddings"].size else embedding
#                 self.index.add(embedding)
#             logger.info(f"Added document {filename} with {len(chunks)} chunks")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to upload document: {str(e)}")
#             return False

#     def search(self, query, k=5, threshold=0.65):
#         """Search knowledge base using FAISS for top-k relevant items."""
#         if not self.knowledge["texts"]:
#             return None

#         query_embedding = self.embedder.encode([query], convert_to_numpy=True)
#         distances, indices = self.index.search(query_embedding, k)
#         results = []

#         for i, idx in enumerate(indices[0]):
#             score = 1 - (distances[0][i] / 2)
#             if score > threshold:
#                 results.append({
#                     "text": self.knowledge["texts"][idx],
#                     "answer": self.knowledge["answers"][idx],
#                     "score": float(score),
#                     "source": self.knowledge["sources"][idx]
#                 })

#         return results if results else None

#     def add_entry(self, question, answer):
#         """Add a new Q&A pair to the knowledge base."""
#         if self.knowledge["texts"]:
#             question_embedding = self.embedder.encode([question], convert_to_numpy=True)
#             distances, _ = self.index.search(question_embedding, 1)
#             if (1 - (distances[0][0] / 2)) > 0.95:
#                 logger.info(f"Skipping duplicate question: {question}")
#                 return False

#         dynamic_file = os.path.join(self.kb_dir, "dynamic_knowledge.txt")
#         try:
#             with open(dynamic_file, 'a', encoding='utf-8') as f:
#                 f.write(f"Q: {question}\nA: {answer}\n\n")
#             self.knowledge["texts"].append(question)
#             self.knowledge["answers"].append(answer)
#             self.knowledge["sources"].append("dynamic_knowledge.txt")
#             embedding = self.embedder.encode([question], convert_to_numpy=True)
#             self.knowledge["embeddings"] = np.vstack(
#                 [self.knowledge["embeddings"], embedding]
#             ) if self.knowledge["embeddings"].size else embedding
#             self.index.add(embedding)
#             logger.info(f"Added new Q&A pair: Q: {question}")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to add Q&A pair: {str(e)}")
#             return False

# class AudioHandler:
#     """Handles all audio operations including STT and TTS."""
#     def __init__(self):
#         self.sample_rate = 16000
#         self._initialize_stt()
#         logger.info("Audio handler initialized")

#     def _initialize_stt(self):
#         """Initialize speech-to-text model."""
#         try:
#             logger.info("Loading Whisper model...")
#             self.stt_model = WhisperModel(
#                 "base",
#                 device="cpu",
#                 compute_type="int8"
#             )
#         except Exception as e:
#             logger.error(f"Failed to load Whisper: {str(e)}")
#             raise RuntimeError("Could not initialize speech recognition")

#     def record_audio(self, duration=5):
#         """Record audio from microphone."""
#         try:
#             logger.info(f"Recording audio for {duration} seconds...")
#             devices = sd.query_devices()
#             logger.debug(f"Available audio devices: {devices}")
#             recording = sd.rec(
#                 int(duration * self.sample_rate),
#                 samplerate=self.sample_rate,
#                 channels=1,
#                 dtype='float32'
#             )
#             sd.wait()
#             return recording.flatten()
#         except Exception as e:
#             logger.error(f"Audio recording failed: {str(e)}")
#             raise RuntimeError(f"Could not record audio: {str(e)}")

#     def transcribe(self, audio_np):
#         """Transcribe audio to text."""
#         try:
#             logger.info("Transcribing audio...")
#             segments, _ = self.stt_model.transcribe(audio_np)
#             full_text = " ".join([segment.text for segment in segments])
#             logger.debug(f"Transcription: {full_text}")
#             return full_text
#         except Exception as e:
#             logger.error(f"Transcription failed: {str(e)}")
#             raise RuntimeError("Could not transcribe audio")

#     def text_to_speech(self, text):
#         """Convert text to speech using pyttsx3."""
#         try:
#             logger.info(f"Speaking text with pyttsx3: {text[:50]}...")
#             engine = pyttsx3.init()
#             engine.setProperty('rate', 150)
#             engine.setProperty('volume', 0.9)
#             engine.say(text)
#             engine.runAndWait()
#             engine.stop()
#             logger.info("pyttsx3 speech synthesis completed successfully")
#             return True
#         except Exception as e:
#             logger.error(f"pyttsx3 speech synthesis failed: {str(e)}")
#             return False

# class ConversationHistory:
#     """Manages conversation history and session logging."""
#     def __init__(self):
#         self.history_dir = "conversation_history"
#         self.current_session = []
#         self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.context_id = str(uuid.uuid4())

#         if not os.path.exists(self.history_dir):
#             os.makedirs(self.history_dir)
#             logger.info(f"Created conversation history directory at {self.history_dir}")

#     def add_entry(self, query, response, mode, feedback=None):
#         """Add an entry to the conversation history."""
#         entry = {
#             "timestamp": datetime.now().isoformat(),
#             "query": query,
#             "response": response,
#             "mode": mode,
#             "feedback": feedback,
#             "context_id": self.context_id
#         }
#         self.current_session.append(entry)
#         logger.debug(f"Added conversation entry: {entry}")

#     def save_session(self):
#         """Save current session to file."""
#         if not self.current_session:
#             logger.warning("No conversation history to save")
#             return False

#         filename = os.path.join(
#             self.history_dir,
#             f"session_{self.session_id}.json"
#         )

#         try:
#             with open(filename, 'w', encoding='utf-8') as f:
#                 json.dump(self.current_session, f, indent=2)
#             logger.info(f"Saved conversation history to {filename}")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to save history: {str(e)}")
#             return False

#     def get_recent_context(self, max_entries=3):
#         """Get recent conversation entries for context."""
#         return self.current_session[-max_entries:] if self.current_session else []

#     def format_context_for_prompt(self, max_entries=3):
#         """Format recent conversation entries for LLM prompt."""
#         recent_context = self.get_recent_context(max_entries)
#         if not recent_context:
#             logger.debug("No recent conversation context available")
#             return ""

#         context_str = "Conversation history:\n"
#         for entry in recent_context:
#             context_str += f"User: {entry['query']}\nAssistant: {entry['response']}\n"
#         logger.debug(f"Formatted context:\n{context_str}")
#         return context_str

# class FeedbackHandler:
#     """Handles user feedback for continuous learning."""
#     def __init__(self, knowledge_base):
#         self.knowledge_base = knowledge_base
#         self.feedback_dir = "feedback_logs"
#         if not os.path.exists(self.feedback_dir):
#             os.makedirs(self.feedback_dir)
#             logger.info(f"Created feedback logs directory at {self.feedback_dir}")

#     def log_feedback(self, query, response, feedback_score, feedback_text=None):
#         """Log user feedback and update knowledge base if positive."""
#         feedback_entry = {
#             "timestamp": datetime.now().isoformat(),
#             "query": query,
#             "response": response,
#             "feedback_score": feedback_score,
#             "feedback_text": feedback_text
#         }

#         feedback_file = os.path.join(self.feedback_dir, f"feedback_{uuid.uuid4()}.json")
#         try:
#             with open(feedback_file, 'w', encoding='utf-8') as f:
#                 json.dump(feedback_entry, f, indent=2)
#             logger.info(f"Saved feedback to {feedback_file}")
#         except Exception as e:
#             logger.error(f"Failed to save feedback: {str(e)}")

#         if feedback_score >= 4:
#             self.knowledge_base.add_entry(query, response)
#             logger.info(f"Positive feedback received; added Q&A to knowledge base")
#         elif feedback_score <= 2 and feedback_text:
#             logger.info(f"Low feedback score; flagged for review: {feedback_text}")

# class AIAgent:
#     """Main AI agent class combining all components."""
#     def __init__(self):
#         self.knowledge_base = KnowledgeBase()
#         self.conversation = ConversationHistory()
#         self.audio = AudioHandler()
#         self.feedback_handler = FeedbackHandler(self.knowledge_base)
#         self._initialize_llm()
#         self.response_cache = {}
#         logger.info("AI Agent initialized successfully")

#     def _initialize_llm(self):
#         """Initialize the Together AI client."""
#         try:
#             api_key = os.getenv("TOGETHER_API_KEY")
#             if not api_key:
#                 raise ValueError("TOGETHER_API_KEY not found in .env")

#             self.llm_client = Together(api_key=api_key)
#             try:
#                 self.llm_client.chat.completions.create(
#                     model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
#                     messages=[{"role": "user", "content": "Test"}],
#                     max_tokens=250
#                 )
#                 logger.info("LLM client initialized and connectivity verified")
#             except Exception as e:
#                 logger.warning(f"LLM connectivity test failed: {str(e)}. Will attempt to proceed.")
#         except Exception as e:
#             logger.error(f"Failed to initialize LLM: {str(e)}")
#             raise RuntimeError("Could not initialize language model")

#     def preprocess_query(self, query: str) -> str:
#         """Clean and normalize user query, preserving proper nouns."""
#         query = query.strip()
#         filler_words = ['um', 'uh', 'you know']
#         for filler in filler_words:
#             query = re.sub(rf'\b{filler}\b', '', query, flags=re.IGNORECASE)
#         query = re.sub(r'\s+', ' ', query).strip()
#         logger.debug(f"Preprocessed query: {query}")
#         return query

#     def generate_response(self, query: str, mode: str = "text") -> str:
#         """Generate response using RAG with knowledge base and LLM."""
#         try:
#             original_query = query
#             query = self.preprocess_query(query)
#             logger.info(f"Processing query: {original_query} (preprocessed: {query})")

#             # Check conversation context for location-specific queries
#             context = self.conversation.get_recent_context(max_entries=3)
#             locations = ["Kileleshwa", "Lavington", "Kilimani"]
#             is_location_specific = any(loc.lower() in query.lower() for loc in locations) or \
#                                   any(any(loc.lower() in entry["query"].lower() for loc in locations) for entry in context)

#             kb_results = self.knowledge_base.search(query, k=5, threshold=0.65)
#             context_str = self.conversation.format_context_for_prompt(max_entries=3)

#             if kb_results:
#                 for result in kb_results:
#                     if result["answer"]:
#                         response = result["answer"]
#                         logger.debug(f"Response from Q&A (score: {result['score']}, source: {result['source']})")
#                         self.conversation.add_entry(original_query, response, mode)
#                         return response

#                 context_chunks = [result["text"] for result in kb_results]
#                 response = self._generate_llm_response(query, context_str, context_chunks, is_location_specific)
#                 logger.info(f"Response from RAG: {response[:50]}...")
#                 self.conversation.add_entry(original_query, response, mode)
#                 return response

#             logger.info("No relevant knowledge base results; falling back to LLM")
#             response = self._generate_llm_response(query, context_str, [], is_location_specific)
#             self.conversation.add_entry(original_query, response, mode)
#             logger.info(f"Final response: {response[:50]}...")
#             return response

#         except Exception as e:
#             error_msg = f"Sorry, I couldn't process your request due to an error: {str(e)}. Please try again or contact our agency."
#             logger.error(f"Response generation failed: {str(e)}")
#             self.conversation.add_entry(original_query, error_msg, mode)
#             return error_msg

#     @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=5)
#     def _generate_llm_response(self, query: str, context: str, context_chunks: list, is_location_specific: bool, retries: int = 2) -> str:
#         """Generate response using LLM with context and document chunks, retrying for incomplete responses."""
#         cache_key = f"{query}:{context[:50]}"
#         if cache_key in self.response_cache:
#             logger.info(f"Cache hit for query: {query}")
#             return self.response_cache[cache_key]
#         logger.info(f"Cache miss for query: {query}")

#         greeting_keywords = ['hi', 'hello', 'hey', 'greetings']
#         if query.lower() in greeting_keywords:
#             logger.info("Detected greeting query")
#             response = "Hello! How can I assist you with your real estate needs today?"
#             self.response_cache[cache_key] = response
#             return response

#         max_chunk_length = 200
#         summarized_chunks = []
#         for chunk in context_chunks:
#             if len(chunk.split()) > max_chunk_length:
#                 summary = self.knowledge_base.summarizer(
#                     chunk, max_length=100, min_length=30, do_sample=False
#                 )[0]["summary_text"]
#                 summarized_chunks.append(summary)
#             else:
#                 summarized_chunks.append(chunk)

#         chunk_context = "\n".join([f"Document: {chunk}" for chunk in summarized_chunks])
#         # Add pricing data for location-specific queries
#         additional_context = ""
#         if is_location_specific and "price" in query.lower():
#             additional_context = """
# Pricing data for Nairobi, Kenya (Kileleshwa, Lavington, Kilimani):
# - Studio apartments: KES 35,000 to KES 70,000 per month
# - One-bedroom apartments: KES 45,000 to KES 120,000 per month
# Source: Recent listings from trusted real estate platforms
# """

#         prompt = f"""
# You are a professional real estate agent. Answer the query '{query}' with a concise, user-friendly response in a conversational tone, using present tense. Use the provided document chunks, conversation history, and additional pricing data (if applicable) to inform your response. If the query mentions a person, assume they are a real estate professional or client unless specified otherwise. For indirect or vague queries, infer intent based on the context and provide relevant real estate advice. If no relevant information is available, offer general real estate advice or ask for clarification. Do not include reasoning, <think> tags, or phrases like 'I should,' 'no information,' 'it seems like,' 'Alright, the user,' or 'let me.' Keep responses under 80 words. Ensure the response is complete and includes all requested information (e.g., price ranges).

# Document chunks:
# {chunk_context}

# Conversation history:
# {context}

# Additional context:
# {additional_context}

# Response:
# """
#         logger.debug(f"Constructed LLM prompt:\n{prompt}")

#         for attempt in range(retries + 1):
#             try:
#                 response = self.llm_client.chat.completions.create(
#                     model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
#                     messages=[{"role": "user", "content": prompt}],
#                     max_tokens=400,  # Increased to ensure complete responses
#                     temperature=0.6,
#                     top_p=0.9
#                 )
#                 raw_response = response.choices[0].message.content
#                 logger.info(f"Raw LLM response: {raw_response[:50]}...")

#                 cleaned_response = self._clean_response(raw_response)
#                 invalid_phrases = [
#                     "not enough information", "haven't provided", "i should",
#                     "think", "alright, the user", "it seems like", "let me",
#                     "maybe", "let's say", "from what i know", "i need to",
#                     "okay, the user", "first, i", "next, i"
#                 ]
#                 # Check if response is incomplete (e.g., ends abruptly or lacks key info like price)
#                 is_incomplete = (
#                     len(cleaned_response.split()) < 10 or
#                     ("price" in query.lower() and not re.search(r'KES|KSh|\d+', cleaned_response)) or
#                     cleaned_response.endswith(("from", "to", "range", "starts"))
#                 )
#                 if (is_incomplete or
#                         any(phrase in cleaned_response.lower() for phrase in invalid_phrases) or
#                         re.search(r'<\s*think\s*>|<\s*/\s*think\s*>', cleaned_response, re.IGNORECASE)):
#                     logger.info(f"Response incomplete or contains invalid phrases (attempt {attempt + 1}); retrying or using fallback")
#                     if attempt == retries:
#                         cleaned_response = self._fetch_external_knowledge(query, context, is_location_specific)
#                 else:
#                     self.response_cache[cache_key] = cleaned_response
#                     return cleaned_response

#             except requests.exceptions.HTTPError as e:
#                 logger.error(f"HTTP error: {e.response.status_code}, {e.response.text}")
#                 if e.response.status_code == 429:
#                     wait_time = 10 * (2 ** self._generate_llm_response.backoff_count)
#                     logger.warning(f"Rate limit hit, waiting {wait_time}s")
#                     time.sleep(wait_time)
#                 if attempt == retries:
#                     return self._fetch_external_knowledge(query, context, is_location_specific)
#             except Exception as e:
#                 logger.error(f"LLM query failed: {str(e)}")
#                 if attempt == retries:
#                     return self._fetch_external_knowledge(query, context, is_location_specific)

#         return self._fetch_external_knowledge(query, context, is_location_specific)

#     def _fetch_external_knowledge(self, query: str, context: str, is_location_specific: bool) -> str:
#         """Fallback to knowledge base with lower threshold or external pricing data."""
#         logger.info(f"Fetching fallback knowledge for query: {query}")
        
#         # Try knowledge base with a lower threshold
#         kb_results = self.knowledge_base.search(query, k=3, threshold=0.5)
#         if kb_results:
#             context_chunks = [result["text"] for result in kb_results]
#             chunk_context = "\n".join([f"Document: {chunk}" for chunk in context_chunks])
#             additional_context = ""
#             if is_location_specific and "price" in query.lower():
#                 additional_context = """
# Pricing data for Nairobi, Kenya (Kileleshwa, Lavington, Kilimani):
# - Studio apartments: KES 35,000 to KES 70,000 per month
# - One-bedroom apartments: KES 45,000 to KES 120,000 per month
# Source: Recent listings from trusted real estate platforms
# """
#             prompt = f"""
# You are a professional real estate agent. Answer the query '{query}' with a concise, user-friendly response in a conversational tone, using present tense. Use the provided document chunks, conversation history, and additional pricing data (if applicable) to inform your response. If the query is vague, infer intent and provide relevant real estate advice. Do not include reasoning or phrases like 'no information,' 'I should,' 'it seems like,' 'Alright, the user,' or 'let me.' Keep responses under 80 words. Ensure the response is complete and includes all requested information (e.g., price ranges).

# Document chunks:
# {chunk_context}

# Conversation history:
# {context}

# Additional context:
# {additional_context}

# Response:
# """
#             try:
#                 response = self.llm_client.chat.completions.create(
#                     model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
#                     messages=[{"role": "user", "content": prompt}],
#                     max_tokens=400,
#                     temperature=0.6,
#                     top_p=0.9
#                 )
#                 return self._clean_response(response.choices[0].message.content)
#             except Exception as e:
#                 logger.error(f"LLM fallback failed: {str(e)}")

#         # Ultimate fallback with pricing data if location-specific
#         if is_location_specific and "price" in query.lower():
#             return "In Kileleshwa, Lavington, or Kilimani, studio apartments rent for KES 35,000 to KES 70,000 per month, and one-bedroom apartments range from KES 45,000 to KES 120,000 per month."
#         return "I can help with real estate questions. Please clarify your query or ask about properties, market trends, or buying processes."

#     def _clean_response(self, text: str) -> str:
#         """Clean LLM response to remove unwanted formatting, reasoning, and preserve numerical data."""
#         # Remove <think> tags and their contents
#         cleaned_text = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
#         cleaned_text = re.sub(r'<\s*think\s*>|<\s*/\s*think\s*>', '', cleaned_text, flags=re.IGNORECASE)
        
#         # Remove reasoning phrases and conversational fluff
#         reasoning_phrases = [
#             r'Alright, the user', r'I should', r'Let me put that together', r'From what I know',
#             r'it seems like', r'let me', r'okay, the user', r'let\'s say', r'maybe ask',
#             r'first, i', r'next, i', r'I need to', r'Wait, the user', r'So, they\'re referring'
#         ]
#         for phrase in reasoning_phrases:
#             cleaned_text = re.sub(phrase + r'.*?(?=\n|$)', '', cleaned_text, flags=re.IGNORECASE)
        
#         # Remove additional conversational or instructional phrases
#         cleaned_text = re.sub(r'(I\'m going to|Here\'s what I think|Based on that).*?(?=\n|$)', '', cleaned_text, flags=re.IGNORECASE)
        
#         # Preserve numerical data (e.g., prices)
#         cleaned_text = cleaned_text.strip().replace('\n', ' ').replace('\r', ' ')
#         cleaned_text = re.sub(r'```|\*\*|#{2,}', '', cleaned_text)
#         cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
#         cleaned_text = cleaned_text.strip('"').strip("'").strip()
        
#         # Ensure price ranges are not cut off
#         if re.search(r'KES|KSh|\d+', cleaned_text) and cleaned_text.endswith(("from", "to", "range", "starts")):
#             cleaned_text = cleaned_text.rsplit(' ', 1)[0] + " (complete range not provided)."
        
#         logger.debug(f"Cleaned response: {cleaned_text}")
#         return cleaned_text

#     def upload_document(self, file_path):
#         """Upload a document to the knowledge base."""
#         return self.knowledge_base.upload_document(file_path)

#     def process_feedback(self, query: str, response: str, feedback_score: int, feedback_text: str = None):
#         """Process user feedback to improve agent performance."""
#         self.feedback_handler.log_feedback(query, response, feedback_score, feedback_text)

# def main():
#     """Run test cases for the AI agent."""
#     print("Initializing AI Agent...")
#     agent = AIAgent()

#     # Test document upload (replace with actual path)
#     test_doc = "test_document.pdf"
#     if os.path.exists(test_doc):
#         print(f"\nUploading test document: {test_doc}")
#         success = agent.upload_document(test_doc)
#         print(f"Document upload {'successful' if success else 'failed'}")

#     test_queries = [
#         "how can i verify documents using current tech on real estate",
#         "mention 2 key trends in real-estate",
#         "how can i use ai in real estate, mention 2 ways",
#         "is it possible to have a virtual walk through of a property using ai",
#         "name one tool i can use to do that",
#         "who founded matterport",
#         "ok give me tips related to that",
#         "am looking for a nice cosy house to rent in kenya, am alone and dont really need much space. what kind of house would you recommend",
#         "what is the price range"
#     ]

#     for test_query in test_queries:
#         print(f"\nTest query: {test_query}")
#         response = agent.generate_response(test_query)
#         print(f"Response: {response}")

#     try:
#         print("\nTesting voice recording (6 seconds)...")
#         audio = agent.audio.record_audio(6)
#         transcription = agent.audio.transcribe(audio)
#         print(f"Transcription: {transcription}")

#         print("Testing response to transcribed query...")
#         response = agent.generate_response(transcription, mode="voice")
#         print(f"Response: {response}")

#         print("Testing TTS...")
#         success = agent.audio.text_to_speech(response)
#         print(f"TTS test {'successful' if success else 'failed'}")
#     except Exception as e:
#         print(f"Voice test failed: {str(e)}")

#     print("\nTesting feedback mechanism...")
#     agent.process_feedback(test_queries[0], "I can help with real estate questions.", 4, "Very helpful!")
#     print("Feedback test complete")

#     agent.conversation.save_session()
#     print("\nAgent test complete")

# # Commenting out main() execution for production
# if __name__ == "__main__":
#      main()



import json
import logging
import os
import re
import time
import uuid
from datetime import datetime

import backoff
import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from together import Together
import pyttsx3
import pdfplumber
import faiss
from transformers import pipeline

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter to suppress torch.classes warning
class TorchClassesFilter(logging.Filter):
    def filter(self, record):
        return not ("Examining the path of torch.classes raised" in record.getMessage())

logging.getLogger().addFilter(TorchClassesFilter())

# Load environment variables
load_dotenv()

class KnowledgeBase:
    """Handles loading, searching, and updating knowledge base with Q&A and documents."""
    def __init__(self, kb_dir="knowledge_base", chunk_size=300):
        self.kb_dir = kb_dir
        self.chunk_size = chunk_size
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.knowledge = {
            "texts": [],
            "answers": [],
            "sources": [],
            "embeddings": []
        }
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self._load_knowledge()

    def _chunk_text(self, text):
        """Split text into chunks of approximately chunk_size words."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += 1
            if current_length >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _load_knowledge(self):
        """Load Q&A files and documents from directory."""
        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir)
            logger.warning(f"Created empty knowledge base directory at {self.kb_dir}")
            return

        for filename in os.listdir(self.kb_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.kb_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                current_q = None
                for line in lines:
                    line = line.strip()
                    if line.startswith("Q:"):
                        current_q = line[3:].strip()
                    elif line.startswith("A:") and current_q:
                        self.knowledge["texts"].append(current_q)
                        self.knowledge["answers"].append(line[3:].strip())
                        self.knowledge["sources"].append(filename)
                        current_q = None

            elif filename.endswith(".pdf"):
                filepath = os.path.join(self.kb_dir, filename)
                with pdfplumber.open(filepath) as pdf:
                    text = " ".join(page.extract_text() or "" for page in pdf.pages)
                chunks = self._chunk_text(text)
                for chunk in chunks:
                    self.knowledge["texts"].append(chunk)
                    self.knowledge["answers"].append(None)
                    self.knowledge["sources"].append(filename)

        if self.knowledge["texts"]:
            logger.info(f"Loaded {len(self.knowledge['texts'])} items (Q&A + document chunks)")
            self.knowledge["embeddings"] = self.embedder.encode(
                self.knowledge["texts"], convert_to_numpy=True
            )
            self.index.add(self.knowledge["embeddings"])
        else:
            logger.warning("No valid data found in knowledge base")

    def upload_document(self, file_path):
        """Process and add a new document to the knowledge base."""
        try:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.kb_dir, filename)
            os.rename(file_path, dest_path)

            if filename.endswith(".pdf"):
                with pdfplumber.open(dest_path) as pdf:
                    text = " ".join(page.extract_text() or "" for page in pdf.pages)
                chunks = self._chunk_text(text)
            elif filename.endswith(".txt"):
                with open(dest_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunks = self._chunk_text(text)
            else:
                logger.error(f"Unsupported file type: {filename}")
                return False

            for chunk in chunks:
                self.knowledge["texts"].append(chunk)
                self.knowledge["answers"].append(None)
                self.knowledge["sources"].append(filename)
                embedding = self.embedder.encode([chunk], convert_to_numpy=True)
                self.knowledge["embeddings"] = np.vstack(
                    [self.knowledge["embeddings"], embedding]
                ) if self.knowledge["embeddings"].size else embedding
                self.index.add(embedding)
            logger.info(f"Added document {filename} with {len(chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to upload document: {str(e)}")
            return False

    def search(self, query, k=5, threshold=0.65):
        """Search knowledge base using FAISS for top-k relevant items."""
        if not self.knowledge["texts"]:
            return None

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        results = []

        # Adaptive threshold based on query length and type
        query_length = len(query.split())
        adjusted_threshold = min(0.75, threshold + (0.05 if query_length > 10 else -0.05))
        if "price" in query.lower():
            adjusted_threshold -= 0.1  # Lower threshold for price queries to include more results

        for i, idx in enumerate(indices[0]):
            score = 1 - (distances[0][i] / 2)
            if score > adjusted_threshold:
                results.append({
                    "text": self.knowledge["texts"][idx],
                    "answer": self.knowledge["answers"][idx],
                    "score": float(score),
                    "source": self.knowledge["sources"][idx]
                })

        return results if results else None

    def add_entry(self, question, answer):
        """Add a new Q&A pair to the knowledge base."""
        if self.knowledge["texts"]:
            question_embedding = self.embedder.encode([question], convert_to_numpy=True)
            distances, _ = self.index.search(question_embedding, 1)
            if (1 - (distances[0][0] / 2)) > 0.95:
                logger.info(f"Skipping duplicate question: {question}")
                return False

        dynamic_file = os.path.join(self.kb_dir, "dynamic_knowledge.txt")
        try:
            with open(dynamic_file, 'a', encoding='utf-8') as f:
                f.write(f"Q: {question}\nA: {answer}\n\n")
            self.knowledge["texts"].append(question)
            self.knowledge["answers"].append(answer)
            self.knowledge["sources"].append("dynamic_knowledge.txt")
            embedding = self.embedder.encode([question], convert_to_numpy=True)
            self.knowledge["embeddings"] = np.vstack(
                [self.knowledge["embeddings"], embedding]
            ) if self.knowledge["embeddings"].size else embedding
            self.index.add(embedding)
            logger.info(f"Added new Q&A pair: Q: {question}")
            return True
        except Exception as e:
            logger.error(f"Failed to add Q&A pair: {str(e)}")
            return False

class AudioHandler:
    """Handles all audio operations including STT and TTS."""
    def __init__(self):
        self.sample_rate = 16000
        self._initialize_stt()
        logger.info("Audio handler initialized")

    def _initialize_stt(self):
        """Initialize speech-to-text model."""
        try:
            logger.info("Loading Whisper model...")
            self.stt_model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8"
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper: {str(e)}")
            raise RuntimeError("Could not initialize speech recognition")

    def record_audio(self, duration=5):
        """Record audio from microphone."""
        try:
            logger.info(f"Recording audio for {duration} seconds...")
            devices = sd.query_devices()
            logger.debug(f"Available audio devices: {devices}")
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            return recording.flatten()
        except Exception as e:
            logger.error(f"Audio recording failed: {str(e)}")
            raise RuntimeError(f"Could not record audio: {str(e)}")

    def transcribe(self, audio_np):
        """Transcribe audio to text."""
        try:
            logger.info("Transcribing audio...")
            segments, _ = self.stt_model.transcribe(audio_np)
            full_text = " ".join([segment.text for segment in segments])
            logger.debug(f"Transcription: {full_text}")
            return full_text
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError("Could not transcribe audio")

    def text_to_speech(self, text):
        """Convert text to speech using pyttsx3."""
        try:
            logger.info(f"Speaking text with pyttsx3: {text[:50]}...")
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            logger.info("pyttsx3 speech synthesis completed successfully")
            return True
        except Exception as e:
            logger.error(f"pyttsx3 speech synthesis failed: {str(e)}")
            return False

class ConversationHistory:
    """Manages conversation history and session logging."""
    def __init__(self):
        self.history_dir = "conversation_history"
        self.current_session = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.context_id = str(uuid.uuid4())

        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
            logger.info(f"Created conversation history directory at {self.history_dir}")

    def add_entry(self, query, response, mode, feedback=None):
        """Add an entry to the conversation history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "mode": mode,
            "feedback": feedback,
            "context_id": self.context_id
        }
        self.current_session.append(entry)
        logger.debug(f"Added conversation entry: {entry}")

    def save_session(self):
        """Save current session to file."""
        if not self.current_session:
            logger.warning("No conversation history to save")
            return False

        filename = os.path.join(
            self.history_dir,
            f"session_{self.session_id}.json"
        )

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, indent=2)
            logger.info(f"Saved conversation history to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save history: {str(e)}")
            return False

    def get_recent_context(self, max_entries=3):
        """Get recent conversation entries for context."""
        return self.current_session[-max_entries:] if self.current_session else []

    def format_context_for_prompt(self, max_entries=3):
        """Format recent conversation entries for LLM prompt."""
        recent_context = self.get_recent_context(max_entries)
        if not recent_context:
            logger.debug("No recent conversation context available")
            return ""

        context_str = "Conversation history:\n"
        for entry in recent_context:
            context_str += f"User: {entry['query']}\nAssistant: {entry['response']}\n"
        logger.debug(f"Formatted context:\n{context_str}")
        return context_str

class FeedbackHandler:
    """Handles user feedback for continuous learning."""
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.feedback_dir = "feedback_logs"
        if not os.path.exists(self.feedback_dir):
            os.makedirs(self.feedback_dir)
            logger.info(f"Created feedback logs directory at {self.feedback_dir}")

    def log_feedback(self, query, response, feedback_score, feedback_text=None):
        """Log user feedback and update knowledge base if positive."""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "feedback_score": feedback_score,
            "feedback_text": feedback_text
        }

        feedback_file = os.path.join(self.feedback_dir, f"feedback_{uuid.uuid4()}.json")
        try:
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_entry, f, indent=2)
            logger.info(f"Saved feedback to {feedback_file}")
        except Exception as e:
            logger.error(f"Failed to save feedback: {str(e)}")

        if feedback_score >= 4:
            self.knowledge_base.add_entry(query, response)
            logger.info(f"Positive feedback received; added Q&A to knowledge base")
        elif feedback_score <= 2 and feedback_text:
            logger.info(f"Low feedback score; flagged for review: {feedback_text}")

class AIAgent:
    """Main AI agent class combining all components."""
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.conversation = ConversationHistory()
        self.audio = AudioHandler()
        self.feedback_handler = FeedbackHandler(self.knowledge_base)
        self._initialize_llm()
        self.response_cache = {}
        logger.info("AI Agent initialized successfully")

    def _initialize_llm(self):
        """Initialize the Together AI client."""
        try:
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY not found in .env")

            self.llm_client = Together(api_key=api_key)
            # Test connectivity
            self.llm_client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=250
            )
            logger.info("LLM client initialized and connectivity verified")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise RuntimeError("Could not initialize language model")

    def preprocess_query(self, query: str) -> str:
        """Clean and normalize user query, preserving proper nouns."""
        query = query.strip()
        filler_words = ['um', 'uh', 'you know']
        for filler in filler_words:
            query = re.sub(rf'\b{filler}\b', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+', ' ', query).strip()
        logger.debug(f"Preprocessed query: {query}")
        return query

    def generate_response(self, query: str, mode: str = "text") -> str:
        """Generate response using RAG with knowledge base and LLM."""
        try:
            original_query = query
            query = self.preprocess_query(query)
            logger.info(f"Processing query: {original_query} (preprocessed: {query})")

            # Enhanced location detection
            context = self.conversation.get_recent_context(max_entries=3)
            locations = ["Kileleshwa", "Lavington", "Kilimani"]
            is_location_specific = (
                any(loc.lower() in query.lower() for loc in locations) or
                any(any(loc.lower() in entry["query"].lower() or loc.lower() in entry["response"].lower() for loc in locations) for entry in context)
            )

            kb_results = self.knowledge_base.search(query, k=5, threshold=0.65)
            context_str = self.conversation.format_context_for_prompt(max_entries=3)

            if kb_results:
                for result in kb_results:
                    if result["answer"]:
                        response = result["answer"]
                        logger.debug(f"Response from Q&A (score: {result['score']}, source: {result['source']})")
                        self.conversation.add_entry(original_query, response, mode)
                        return response

                context_chunks = [result["text"] for result in kb_results]
                response = self._generate_llm_response(query, context_str, context_chunks, is_location_specific)
                logger.info(f"Response from RAG: {response[:50]}...")
                self.conversation.add_entry(original_query, response, mode)
                return response

            logger.info("No relevant knowledge base results; falling back to LLM")
            response = self._generate_llm_response(query, context_str, [], is_location_specific)
            self.conversation.add_entry(original_query, response, mode)
            logger.info(f"Final response: {response[:50]}...")
            return response

        except Exception as e:
            error_msg = f"Sorry, I couldn't process your request due to an error: {str(e)}. Please try again or contact our agency."
            logger.error(f"Response generation failed: {str(e)}")
            self.conversation.add_entry(original_query, error_msg, mode)
            return error_msg

    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=5)
    def _generate_llm_response(self, query: str, context: str, context_chunks: list, is_location_specific: bool, retries: int = 2) -> str:
        """Generate response using LLM with context and document chunks, retrying for incomplete responses."""
        cache_key = f"{query}:{context[:50]}"
        if cache_key in self.response_cache:
            logger.info(f"Cache hit for query: {query}")
            return self.response_cache[cache_key]
        logger.info(f"Cache miss for query: {query}")

        greeting_keywords = ['hi', 'hello', 'hey', 'greetings']
        if query.lower() in greeting_keywords:
            logger.info("Detected greeting query")
            response = "Hello! How can I assist you with your real estate needs today?"
            self.response_cache[cache_key] = response
            return response

        max_chunk_length = 200
        summarized_chunks = []
        for chunk in context_chunks:
            if len(chunk.split()) > max_chunk_length:
                summary = self.knowledge_base.summarizer(
                    chunk, max_length=100, min_length=30, do_sample=False
                )[0]["summary_text"]
                summarized_chunks.append(summary)
            else:
                summarized_chunks.append(chunk)

        chunk_context = "\n".join([f"Document: {chunk}" for chunk in summarized_chunks])
        additional_context = ""
        if is_location_specific and "price" in query.lower():
            additional_context = """
Pricing data for Nairobi, Kenya (Kileleshwa, Lavington, Kilimani):
- Studio apartments: KES 35,000 to KES 70,000 per month
- One-bedroom apartments: KES 45,000 to KES 120,000 per month
Source: Recent listings from trusted real estate platforms
"""

        prompt = f"""
You are a professional real estate agent with expertise in Nairobi, Kenya. Answer the query '{query}' with a concise, accurate, and user-friendly response in a conversational tone, using present tense. Leverage the provided document chunks, conversation history, and additional pricing data (if applicable) to ensure relevance. For vague or indirect queries, infer intent based on context and provide actionable real estate advice. If the query mentions a person, assume they are a real estate professional or client. Ensure responses are complete, especially for price-related queries (include specific ranges, e.g., KES 35,000-70,000). Avoid speculative language, reasoning, or phrases like 'I should,' 'it seems like,' 'no information,' 'alright,' or 'let me.' Keep responses under 80 words and fact-based.

Document chunks:
{chunk_context}

Conversation history:
{context}

Additional context:
{additional_context}

Response:
"""
        logger.debug(f"Constructed LLM prompt:\n{prompt}")

        for attempt in range(retries + 1):
            try:
                response = self.llm_client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.6,
                    top_p=0.9
                )
                raw_response = response.choices[0].message.content
                logger.info(f"Raw LLM response: {raw_response[:50]}...")

                cleaned_response = self._clean_response(raw_response, query)
                invalid_phrases = [
                    "not enough information", "haven't provided", "i should",
                    "think", "alright, the user", "it seems like", "let me",
                    "maybe", "let's say", "from what i know", "i need to",
                    "okay, the user", "first, i", "next, i"
                ]
                is_incomplete = (
                    len(cleaned_response.split()) < 10 or
                    ("price" in query.lower() and not re.search(r'KES|KSh|\d+', cleaned_response)) or
                    cleaned_response.endswith(("from", "to", "range", "starts"))
                )
                if (is_incomplete or
                        any(phrase in cleaned_response.lower() for phrase in invalid_phrases) or
                        re.search(r'<\s*think\s*>|<\s*/\s*think\s*>', cleaned_response, re.IGNORECASE)):
                    logger.info(f"Response incomplete or contains invalid phrases (attempt {attempt + 1}); retrying or using fallback")
                    if attempt == retries:
                        cleaned_response = self._fetch_external_knowledge(query, context, is_location_specific)
                else:
                    if len(cleaned_response.split()) >= 5 and not re.search(r'\(complete range not provided\)', cleaned_response):
                        self.response_cache[cache_key] = cleaned_response
                    return cleaned_response

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e.response.status_code}, {e.response.text}")
                if e.response.status_code == 429:
                    wait_time = 10 * (2 ** self._generate_llm_response.backoff_count)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s")
                    time.sleep(wait_time)
                if attempt == retries:
                    return self._fetch_external_knowledge(query, context, is_location_specific)
            except Exception as e:
                logger.error(f"LLM query failed: {str(e)}")
                if attempt == retries:
                    return self._fetch_external_knowledge(query, context, is_location_specific)

        return self._fetch_external_knowledge(query, context, is_location_specific)

    def _fetch_external_knowledge(self, query: str, context: str, is_location_specific: bool) -> str:
        """Fallback to knowledge base with lower threshold or predefined templates."""
        logger.info(f"Fetching fallback knowledge for query: {query}")

        threshold = 0.4 if is_location_specific else 0.5
        kb_results = self.knowledge_base.search(query, k=3, threshold=threshold)
        if kb_results:
            context_chunks = [result["text"] for result in kb_results]
            chunk_context = "\n".join([f"Document: {chunk}" for chunk in context_chunks])
            additional_context = ""
            if is_location_specific and "price" in query.lower():
                additional_context = """
Pricing data for Nairobi, Kenya (Kileleshwa, Lavington, Kilimani):
- Studio apartments: KES 35,000 to KES 70,000 per month
- One-bedroom apartments: KES 45,000 to KES 120,000 per month
Source: Recent listings from trusted real estate platforms
"""
            prompt = f"""
You are a professional real estate agent with expertise in Nairobi, Kenya. Answer the query '{query}' with a concise, accurate, and user-friendly response in a conversational tone, using present tense. Leverage the provided document chunks, conversation history, and additional pricing data (if applicable) to ensure relevance. For vague or indirect queries, infer intent based on context and provide actionable real estate advice. If the query mentions a person, assume they are a real estate professional or client. Ensure responses are complete, especially for price-related queries (include specific ranges, e.g., KES 35,000-70,000). Avoid speculative language, reasoning, or phrases like 'I should,' 'it seems like,' 'no information,' 'alright,' or 'let me.' Keep responses under 80 words and fact-based.

Document chunks:
{chunk_context}

Conversation history:
{context}

Additional context:
{additional_context}

Response:
"""
            try:
                response = self.llm_client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.6,
                    top_p=0.9
                )
                return self._clean_response(response.choices[0].message.content, query)
            except Exception as e:
                logger.error(f"LLM fallback failed: {str(e)}")

        if is_location_specific and "price" in query.lower():
            return "In Kileleshwa, Lavington, or Kilimani, studio apartments rent for KES 35,000 to KES 70,000 per month, and one-bedroom apartments range from KES 45,000 to KES 120,000 per month."
        if "trend" in query.lower():
            return "Key real estate trends include virtual tours for remote viewing and smart home integrations for energy efficiency."
        return "I can help with real estate questions! Please specify your needs, like property types, locations, or market trends."

    def _clean_response(self, text: str, query: str) -> str:
        """Clean LLM response to remove unwanted formatting and ensure quality."""
        cleaned_text = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        cleaned_text = re.sub(r'<\s*think\s*>|<\s*/\s*think\s*>', '', cleaned_text, flags=re.IGNORECASE)

        reasoning_phrases = [
            r'Alright, the user', r'I should', r'Let me put that together', r'From what I know',
            r'it seems like', r'let me', r'okay, the user', r'let\'s say', r'maybe ask',
            r'first, i', r'next, i', r'I need to', r'Wait, the user', r'So, they\'re referring'
        ]
        for phrase in reasoning_phrases:
            cleaned_text = re.sub(phrase + r'.*?(?=\n|$)', '', cleaned_text, flags=re.IGNORECASE)

        cleaned_text = re.sub(r'(I\'m going to|Here\'s what I think|Based on that).*?(?=\n|$)', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip().replace('\n', ' ').replace('\r', ' ')
        cleaned_text = re.sub(r'```|\*\*|#{2,}', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        cleaned_text = cleaned_text.strip('"').strip("'").strip()

        if len(cleaned_text.split()) < 5:
            logger.warning(f"Response too short: {cleaned_text}")
            return "Please clarify your query for a detailed response."
        if "price" in query.lower() and not re.search(r'KES|KSh|\d+', cleaned_text):
            logger.warning(f"Price query missing price range: {cleaned_text}")
            cleaned_text += " (e.g., KES 35,000-70,000 for studios in Nairobi)."

        logger.debug(f"Cleaned response: {cleaned_text}")
        return cleaned_text

    def upload_document(self, file_path):
        """Upload a document to the knowledge base."""
        return self.knowledge_base.upload_document(file_path)

    def process_feedback(self, query: str, response: str, feedback_score: int, feedback_text: str = None):
        """Process user feedback to improve agent performance."""
        self.feedback_handler.log_feedback(query, response, feedback_score, feedback_text)

def main():
    """Run test cases for the AI agent."""
    print("Initializing AI Agent...")
    agent = AIAgent()

    test_doc = "test_document.pdf"
    if os.path.exists(test_doc):
        print(f"\nUploading test document: {test_doc}")
        success = agent.upload_document(test_doc)
        print(f"Document upload {'successful' if success else 'failed'}")

    test_queries = [
        "how can i verify documents using current tech on real estate",
        "mention 2 key trends in real-estate",
        "how can i use ai in real estate, mention 2 ways",
        "is it possible to have a virtual walk through of a property using ai",
        "name one tool i can use to do that",
        "who founded matterport",
        "ok give me tips related to that",
        "am looking for a nice cosy house to rent in kenya, am alone and dont really need much space. what kind of house would you recommend",
        "what is the price range"
    ]

    for test_query in test_queries:
        print(f"\nTest query: {test_query}")
        response = agent.generate_response(test_query)
        print(f"Response: {response}")

    try:
        print("\nTesting voice recording (6 seconds)...")
        audio = agent.audio.record_audio(6)
        transcription = agent.audio.transcribe(audio)
        print(f"Transcription: {transcription}")

        print("Testing response to transcribed query...")
        response = agent.generate_response(transcription, mode="voice")
        print(f"Response: {response}")

        print("Testing TTS...")
        success = agent.audio.text_to_speech(response)
        print(f"TTS test {'successful' if success else 'failed'}")
    except Exception as e:
        print(f"Voice test failed: {str(e)}")

    print("\nTesting feedback mechanism...")
    agent.process_feedback(test_queries[0], "I can help with real estate questions.", 4, "Very helpful!")
    print("Feedback test complete")

    agent.conversation.save_session()
    print("\nAgent test complete")

if __name__ == "__main__":
    main()