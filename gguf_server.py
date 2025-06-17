import json
from typing import Optional, Any, Dict, List
import os
import glob
from dataclasses import dataclass, field
from enum import Enum
from flask import Flask, request, jsonify
from functools import wraps
import logging
from datetime import datetime
import base64
import requests
from pathlib import Path
import hashlib

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    print("Error: llama-cpp-python not installed. Please install with: pip install llama-cpp-python")
    LLAMA_CPP_AVAILABLE = False
    Llama = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Environment-specific settings
RENDER_MEMORY_LIMIT = os.environ.get("RENDER_MEMORY_LIMIT", "1GB")
MAX_PROMPT_LENGTH = int(os.environ.get("MAX_PROMPT_LENGTH", "6000"))
MAX_CONTEXT_SIZE = int(os.environ.get("MAX_CONTEXT_SIZE", "8192"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llama_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResponseType(Enum):
    CHAT = "chat"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    CODE = "code"

@dataclass
class ModelFormat:
    chat_format: str
    start_sequence: str
    stop_sequences: List[str]
    system_template: str

@dataclass
class PromptTemplate:
    system_prompt: str
    temperature: float
    max_tokens: int
    stop_sequences: List[str]

class ModelFormats:
    LLAMA2 = ModelFormat(
        chat_format="llama-2",
        start_sequence="[INST]",
        stop_sequences=["[/INST]", "User:", "</s>"],
        system_template="[SYSTEM]{}[/SYSTEM]"
    )
    MISTRAL = ModelFormat(
        chat_format="mistral",
        start_sequence="<|im_start|>",
        stop_sequences=["<|im_end|>", "<|endoftext|>"],
        system_template="<|im_start|>system\n{}<|im_end|>"
    )
    NEURAL_CHAT = ModelFormat(
        chat_format="chatml",
        start_sequence="USER:",
        stop_sequences=["ASSISTANT:", "<|endoftext|>"],
        system_template="SYSTEM: {}\n"
    )
    CODELLAMA = ModelFormat(
        chat_format="llama-2",
        start_sequence="[INST]",
        stop_sequences=["[/INST]", "User:", "</s>"],
        system_template="[SYSTEM]{}[/SYSTEM]"
    )
    TINYLLAMA = ModelFormat(
        chat_format="chatml",
        start_sequence="<|user|>",
        stop_sequences=["<|assistant|>", "<|endoftext|>"],
        system_template="<|system|>{}<|endoftext|>"
    )
    DEFAULT = ModelFormat(
        chat_format="raw",
        start_sequence="",
        stop_sequences=["</s>", "<|endoftext|>"],
        system_template="{}\n\n"
    )

class PromptLibrary:
    # Base mindset - 30 words max
    BASE_MINDSET = """You are Adam, a helpful AI assistant. Be direct, honest, and practical. Focus on providing clear, useful responses while being friendly and conversational."""

    DEFAULT = PromptTemplate(
        system_prompt=f"""{BASE_MINDSET}

You provide clear, informative, and natural responses. Engage in conversation naturally while providing accurate and relevant information. If you're unsure about something, acknowledge it honestly. Keep responses concise but complete. When providing examples, format them using code blocks like ```html```, ```js```, or ```python``` as appropriate for the content.""",
        temperature=0.7,
        max_tokens=2048,  # Reduced from 4096
        stop_sequences=["User:", "\n\n"]
    )

    CODE = PromptTemplate(
        system_prompt=f"""{BASE_MINDSET}

As a programming assistant, provide clear, well-documented code solutions with explanations. Focus on best practices and efficient implementations. When sharing code, use the appropriate language formatting such as ```js``` for JavaScript or ```python``` for Python code.""",
        temperature=0.2,
        max_tokens=2048,  # Reduced from 4096
        stop_sequences=["User:", "\n\n"]
    )

    CREATIVE = PromptTemplate(
        system_prompt=f"""{BASE_MINDSET}

As a creative assistant, generate engaging and imaginative content. Think outside the box while maintaining coherence and quality. When providing examples of creative work, format it using the appropriate language blocks like ```html``` or ```text```.""",
        temperature=0.9,
        max_tokens=2048,  # Reduced from 4096
        stop_sequences=["User:", "\n\n"]
    )

class ModelDownloader:
    DEFAULT_MODEL_INFO = {
        "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "expected_size": 4368438272,  # Approximate size in bytes
        "checksum": None  # Add if available
    }

    def __init__(self):
        self.current_dir = Path.cwd()
        self.models_dir = self.current_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        logger.info(f"Models directory: {self.models_dir}")

    def find_existing_gguf(self) -> Optional[str]:
        """Find existing GGUF files in current directory and models subdirectory"""
        search_patterns = [
            str(self.current_dir / "*.gguf"),
            str(self.models_dir / "*.gguf")
        ]
        
        for pattern in search_patterns:
            gguf_files = glob.glob(pattern)
            if gguf_files:
                logger.info(f"Found existing GGUF files: {gguf_files}")
                # Return the first one found, preferring larger files (likely better quality)
                gguf_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                selected_file = gguf_files[0]
                logger.info(f"Selected GGUF file: {selected_file}")
                return selected_file
        
        logger.info("No existing GGUF files found")
        return None

    def download_with_progress(self, url: str, filepath: str) -> bool:
        """Download file with progress logging"""
        try:
            logger.info(f"Starting download from: {url}")
            logger.info(f"Downloading to: {filepath}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            logger.info(f"Total file size: {total_size / (1024*1024):.2f} MB")
            
            downloaded = 0
            chunk_size = 8192
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 100MB
                        if downloaded % (100 * 1024 * 1024) == 0 or downloaded == total_size:
                            progress = (downloaded / total_size * 100) if total_size > 0 else 0
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded / (1024*1024):.2f} MB)")
            
            logger.info(f"Download completed: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    def download_default_model(self) -> Optional[str]:
        """Download the default Llama 2-7B model"""
        model_info = self.DEFAULT_MODEL_INFO
        model_path = self.models_dir / model_info["filename"]
        
        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            return str(model_path)
        
        logger.info("Downloading default Llama 2-7B GGUF model...")
        
        # Try HuggingFace Hub first
        if HF_HUB_AVAILABLE:
            try:
                logger.info("Using huggingface_hub for download...")
                downloaded_path = hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=model_info["filename"],
                    local_dir=str(self.models_dir),
                    local_dir_use_symlinks=False
                )
                logger.info(f"Successfully downloaded via huggingface_hub: {downloaded_path}")
                return downloaded_path
            except Exception as e:
                logger.warning(f"HuggingFace Hub download failed: {str(e)}")
        
        # Fallback to direct download
        try:
            url = f"https://huggingface.co/{model_info['repo_id']}/resolve/main/{model_info['filename']}"
            if self.download_with_progress(url, str(model_path)):
                return str(model_path)
        except Exception as e:
            logger.error(f"Direct download failed: {str(e)}")
        
        # Alternative mirror or smaller model
        try:
            logger.info("Trying alternative smaller model...")
            alt_filename = "llama-2-7b-chat.Q2_K.gguf"
            alt_path = self.models_dir / alt_filename
            alt_url = f"https://huggingface.co/{model_info['repo_id']}/resolve/main/{alt_filename}"
            
            if self.download_with_progress(alt_url, str(alt_path)):
                logger.info(f"Successfully downloaded alternative model: {alt_path}")
                return str(alt_path)
        except Exception as e:
            logger.error(f"Alternative download failed: {str(e)}")
        
        logger.error("All download attempts failed")
        return None

    def get_model_path(self) -> Optional[str]:
        """Get model path - either existing or newly downloaded"""
        # First check for existing models
        existing_model = self.find_existing_gguf()
        if existing_model:
            return existing_model
        
        # Download default model if none exists
        logger.info("No existing GGUF model found, downloading default model...")
        return self.download_default_model()

def rate_limit(max_requests: int, window_seconds: int):
    def decorator(f):
        requests_log = {}
        @wraps(f)
        def wrapped(*args, **kwargs):
            now = datetime.now().timestamp()
            client_ip = request.remote_addr
            requests_log[client_ip] = [t for t in requests_log.get(client_ip, []) if now - t < window_seconds]
            if len(requests_log.get(client_ip, [])) >= max_requests:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
            requests_log.setdefault(client_ip, []).append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator

class EnhancedResponseGenerator:
    def __init__(self):
        self.llm: Optional[Any] = None
        self.model_format: Optional[ModelFormat] = None
        self.model_path: Optional[str] = None
        self.model_config: Dict[str, Any] = {
            "n_ctx": MAX_CONTEXT_SIZE,  # Use environment variable
            "n_threads": 2,  # Reduced from 4 to save memory
            "verbose": False,
            "repeat_penalty": 1.2,
            "top_p": 0.95,
            "top_k": 40,
            "n_batch": 256,  # Add batch size control
            "use_mlock": False,  # Disable memory locking to save RAM
            "use_mmap": True   # Enable memory mapping for efficiency
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else None
        self.current_model_type = "gguf"
        self.downloader = ModelDownloader()
        self.initialize_model()

    def check_memory_usage(self) -> dict:
        """Check current memory usage"""
        if not PSUTIL_AVAILABLE:
            return {"available": False}
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "available": True,
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident memory
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual memory
                "percent": process.memory_percent()
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return {"available": False, "error": str(e)}

    def calculate_safe_max_tokens(self, input_text: str, max_context: int = None) -> int:
        """Calculate safe max_tokens based on input length"""
        if max_context is None:
            max_context = self.model_config["n_ctx"]
        
        # Rough estimation: 1 token ≈ 4 characters
        estimated_input_tokens = len(input_text) // 4
        
        # Reserve tokens for system prompt and formatting
        system_overhead = 300  # Increased overhead for safety
        
        # Calculate available tokens for generation
        available_tokens = max_context - estimated_input_tokens - system_overhead
        
        # Ensure we don't go below minimum or above reasonable maximum
        safe_tokens = max(256, min(available_tokens, 1536))  # More conservative limits
        
        logger.info(f"Input chars: {len(input_text)}, Estimated tokens: {estimated_input_tokens}, Safe max_tokens: {safe_tokens}")
        
        return safe_tokens

    def detect_model_format(self, model_path: str) -> ModelFormat:
        """Detect model format from model path"""
        model_name = os.path.basename(model_path).lower()
        logger.info(f"Detecting format for model: {model_name}")
        
        if "codellama" in model_name:
            logger.info("Detected CodeLlama model")
            return ModelFormats.CODELLAMA
        elif "llama-2" in model_name or "llama2" in model_name:
            logger.info("Detected Llama 2 model")
            return ModelFormats.LLAMA2
        elif "tinyllama" in model_name:
            logger.info("Detected TinyLlama model")
            return ModelFormats.TINYLLAMA
        elif "mistral" in model_name:
            logger.info("Detected Mistral model")
            return ModelFormats.MISTRAL
        elif "neural-chat" in model_name:
            logger.info("Detected Neural Chat model")
            return ModelFormats.NEURAL_CHAT
        else:
            logger.info(f"Using default format for model: {model_name}")
            return ModelFormats.LLAMA2  # Default to Llama 2 format

    def initialize_model(self):
        """Initialize the model - download if necessary and load"""
        try:
            logger.info("Initializing model...")
            
            if not LLAMA_CPP_AVAILABLE:
                logger.error("llama-cpp-python not installed. Please install with: pip install llama-cpp-python")
                raise ImportError("llama-cpp-python not installed")
            
            # Get model path (download if necessary)
            self.model_path = self.downloader.get_model_path()
            if not self.model_path:
                raise FileNotFoundError("Could not find or download a GGUF model")
            
            logger.info(f"Using model: {self.model_path}")
            
            # Detect model format
            self.model_format = self.detect_model_format(self.model_path)
            
            # Load the model
            self.load_model()
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def load_model(self):
        """Load the GGUF model"""
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading GGUF model: {self.model_path}")
            
            # Configure model parameters
            config = self.model_config.copy()
            config["chat_format"] = self.model_format.chat_format
            
            # Adjust settings based on model type
            model_name = os.path.basename(self.model_path).lower()
            if "codellama" in model_name:
                config.update({
                    "temperature": 0.3,
                    "repeat_penalty": 1.1,
                    "top_k": 50,
                    "top_p": 0.9
                })
            elif "tinyllama" in model_name:
                config.update({
                    "temperature": 0.7,
                    "repeat_penalty": 1.1,
                    "n_ctx": min(2048, MAX_CONTEXT_SIZE)  # Smaller context for tiny models
                })
            
            # Try GPU first, fallback to CPU
            try:
                logger.info("Attempting to load model with GPU acceleration...")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=-1,
                    **config
                )
                logger.info("Model loaded successfully with GPU acceleration")
            except Exception as gpu_error:
                logger.warning(f"GPU loading failed: {str(gpu_error)}. Falling back to CPU.")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=0,
                    **config
                )
                logger.info("Model loaded successfully on CPU")
            
            logger.info(f"Model loaded with chat format: {self.model_format.chat_format}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_prompt_template(self, response_type: str) -> PromptTemplate:
        """Get the appropriate prompt template based on response type"""
        type_mapping = {
            ResponseType.CHAT.value: PromptLibrary.DEFAULT,
            ResponseType.ANALYSIS.value: PromptLibrary.DEFAULT,
            ResponseType.CREATIVE.value: PromptLibrary.CREATIVE,
            ResponseType.CODE.value: PromptLibrary.CODE
        }
        return type_mapping.get(response_type, PromptLibrary.DEFAULT)

    def prepare_chat_messages(self, template: PromptTemplate, user_input: str, response_type: str) -> List[Dict[str, str]]:
        """Prepare chat messages with proper formatting"""
        return [
            {"role": "system", "content": template.system_prompt},
            {"role": "user", "content": user_input}
        ]

    def get_generation_params(self, template: PromptTemplate, response_type: str, input_text: str = "") -> Dict[str, Any]:
        """Get generation parameters based on response type and input length"""
        
        # Calculate safe token limit
        safe_max_tokens = self.calculate_safe_max_tokens(input_text)
        
        params = {
            "max_tokens": min(template.max_tokens, safe_max_tokens),  # Use smaller value
            "temperature": template.temperature,
            "top_p": self.model_config["top_p"],
            "repeat_penalty": self.model_config["repeat_penalty"],
            "stop": self.model_format.stop_sequences if self.model_format else template.stop_sequences
        }

        # Adjust parameters based on response type
        if response_type == "code":
            params["temperature"] = 0.2
        elif response_type == "creative":
            params["temperature"] = 0.9

        # Further reduce tokens if memory is high
        memory_info = self.check_memory_usage()
        if memory_info.get("available") and memory_info.get("percent", 0) > 80:
            params["max_tokens"] = min(params["max_tokens"], 512)
            logger.warning(f"High memory usage ({memory_info['percent']:.1f}%), reducing max_tokens to {params['max_tokens']}")

        return params

    def generate_response(self, user_input: str, response_type: str = "chat") -> str:
        """Generate response using the loaded model"""
        if not user_input:
            return json.dumps({
                "conversation_response": "Please provide some input.",
                "status": "error"
            })

        try:
            if self.llm is None:
                raise ValueError("Model not initialized")

            # Check memory before generation
            memory_info = self.check_memory_usage()
            if memory_info.get("available") and memory_info.get("percent", 0) > 85:
                logger.warning(f"High memory usage: {memory_info}")

            logger.info(f"Generating response for type: {response_type}")
            template = self.get_prompt_template(response_type)
            messages = self.prepare_chat_messages(template, user_input, response_type)
            generation_params = self.get_generation_params(template, response_type, user_input)

            logger.debug(f"Generation parameters: {generation_params}")

            # Generate response with error handling
            try:
                output = self.llm.create_chat_completion(
                    messages=messages,
                    **generation_params
                )
            except Exception as gen_error:
                if "context" in str(gen_error).lower() or "length" in str(gen_error).lower():
                    # Try with even smaller tokens
                    generation_params["max_tokens"] = min(generation_params["max_tokens"], 256)
                    logger.warning(f"Context error, retrying with max_tokens={generation_params['max_tokens']}")
                    output = self.llm.create_chat_completion(
                        messages=messages,
                        **generation_params
                    )
                else:
                    raise gen_error

            if output and "choices" in output and output["choices"]:
                conversation_response = output["choices"][0]["message"]["content"].strip()
                
                # Clean up response
                if self.model_format:
                    for stop_seq in self.model_format.stop_sequences:
                        conversation_response = conversation_response.replace(stop_seq, "")
                
                conversation_response = ' '.join(conversation_response.split()).strip()
                
                logger.info("Response generated successfully")
                
                return json.dumps({
                    "conversation_response": conversation_response,
                    "metadata": {
                        "response_type": response_type,
                        "temperature": generation_params["temperature"],
                        "max_tokens": generation_params["max_tokens"],
                        "model_path": os.path.basename(self.model_path) if self.model_path else None,
                        "model_format": self.model_format.chat_format if self.model_format else None,
                        "persona": "Adam",
                        "input_length": len(user_input)
                    },
                    "status": "success"
                })
            else:
                logger.error("Model generated empty response")
                return json.dumps({
                    "conversation_response": "Model did not generate a valid response.",
                    "status": "error"
                })

        except MemoryError:
            logger.error("Out of memory during generation")
            return json.dumps({
                "conversation_response": "Request too large. Please reduce input size.",
                "status": "error",
                "error_type": "memory_limit"
            })
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            
            # Check if it's a context length error
            if "context" in str(e).lower() or "length" in str(e).lower():
                return json.dumps({
                    "conversation_response": "Input too long for current model context. Please reduce input size.",
                    "status": "error",
                    "error_type": "context_length"
                })
            
            return json.dumps({
                "conversation_response": "An error occurred while processing your request. Please try again with a shorter input.",
                "error": str(e),
                "status": "error"
            })

# Initialize Flask app
app = Flask(__name__)

# Initialize generator
try:
    logger.info("Starting response generator initialization...")
    generator = EnhancedResponseGenerator()
    logger.info("Response generator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize response generator: {str(e)}")
    generator = None

@app.route('/generate_response', methods=['POST'])
@rate_limit(max_requests=30, window_seconds=60)  # Reduced from 60 to 30
def get_response():
    """Generate response endpoint"""
    if generator is None:
        return jsonify({
            "error": "Model not initialized. Please check server logs.",
            "status": "error"
        }), 500

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        is_base64 = data.get('base64', False)
        user_input = data.get('text', '')

        if not user_input:
            return jsonify({"error": "No text provided"}), 400

        if is_base64:
            try:
                user_input = base64.b64decode(user_input).decode('utf-8')
            except Exception as e:
                return jsonify({"error": "Invalid base64 encoding", "details": str(e)}), 400

        # INPUT LENGTH VALIDATION
        if len(user_input) > MAX_PROMPT_LENGTH:
            return jsonify({
                "error": f"Input too long. Maximum allowed: {MAX_PROMPT_LENGTH} characters",
                "max_chars": MAX_PROMPT_LENGTH,
                "current_chars": len(user_input),
                "status": "error"
            }), 400

        response_type = data.get('response_type', 'chat')

        if response_type not in [rt.value for rt in ResponseType]:
            return jsonify({
                "error": f"Invalid response_type. Must be one of: {[rt.value for rt in ResponseType]}"
            }), 400

        logger.info(f"Processing request - Type: {response_type}, Input length: {len(user_input)}")
        
        # Check memory before processing
        memory_info = generator.check_memory_usage()
        if memory_info.get("available") and memory_info.get("percent", 0) > 90:
            return jsonify({
                "error": "Server memory usage too high. Please try again later.",
                "status": "error",
                "error_type": "memory_limit"
            }), 503

        response = generator.generate_response(user_input, response_type)
        return response, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        logger.error(f"Error handling request: {str(e)}")
        return jsonify({
            "error": "An error occurred while processing your request",
            "details": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    memory_info = generator.check_memory_usage() if generator else {"available": False}
    
    model_info = {
        "status": "ok" if generator and generator.llm else "error",
        "model_loaded": generator.llm is not None if generator else False,
        "model_path": os.path.basename(generator.model_path) if generator and generator.model_path else None,
        "model_format": generator.model_format.chat_format if generator and generator.model_format else None,
        "persona": "Adam",
        "memory_info": memory_info,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_context_size": MAX_CONTEXT_SIZE
    }
    
    status_code = 200 if model_info["model_loaded"] else 503
    return jsonify(model_info), status_code

@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    if not generator:
        return jsonify({
            "status": "error",
            "message": "Generator not initialized"
        }), 500
    
    memory_info = generator.check_memory_usage()
    
    return jsonify({
        "status": "ready",
        "persona": "Adam",
        "mindset": PromptLibrary.BASE_MINDSET,
        "model_info": {
            "path": generator.model_path,
            "format": generator.model_format.chat_format if generator.model_format else None,
            "loaded": generator.llm is not None,
            "context_size": generator.model_config["n_ctx"]
        },
        "server_info": {
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "hf_hub_available": HF_HUB_AVAILABLE,
            "psutil_available": PSUTIL_AVAILABLE
        },
        "limits": {
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "max_context_size": MAX_CONTEXT_SIZE
        },
        "memory_info": memory_info
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting Adam AI server on port {port}")
    logger.info(f"Max prompt length: {MAX_PROMPT_LENGTH}")
    logger.info(f"Max context size: {MAX_CONTEXT_SIZE}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
