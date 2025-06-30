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
import tempfile
from io import BytesIO

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

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Environment-specific settings
RENDER_MEMORY_LIMIT = os.environ.get("RENDER_MEMORY_LIMIT", "1GB")
MAX_PROMPT_LENGTH = int(os.environ.get("MAX_PROMPT_LENGTH", "6000"))
MAX_CONTEXT_SIZE = int(os.environ.get("MAX_CONTEXT_SIZE", "8192"))
ENABLE_VISION = os.environ.get("ENABLE_VISION", "true").lower() == "true"

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
    VISION = "vision"

@dataclass
class ModelFormat:
    chat_format: str
    start_sequence: str
    stop_sequences: List[str]
    system_template: str
    supports_vision: bool = False

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
        system_template="[SYSTEM]{}[/SYSTEM]",
        supports_vision=False
    )
    LLAVA = ModelFormat(
        chat_format="llava-1-5",
        start_sequence="USER:",
        stop_sequences=["ASSISTANT:", "<|endoftext|>", "\nUSER:", "\nHuman:", "\nUser:"],
        system_template="SYSTEM: {}\n",
        supports_vision=True
    )
    LLAVA_V16 = ModelFormat(
        chat_format="llava-v1",
        start_sequence="USER:",
        stop_sequences=["ASSISTANT:", "<|endoftext|>", "\nUSER:", "\nHuman:", "\nUser:"],
        system_template="SYSTEM: {}\n",
        supports_vision=True
    )
    MISTRAL = ModelFormat(
        chat_format="mistral",
        start_sequence="<|im_start|>",
        stop_sequences=["<|im_end|>", "<|endoftext|>"],
        system_template="<|im_start|>system\n{}<|im_end|>",
        supports_vision=False
    )
    NEURAL_CHAT = ModelFormat(
        chat_format="chatml",
        start_sequence="USER:",
        stop_sequences=["ASSISTANT:", "<|endoftext|>"],
        system_template="SYSTEM: {}\n",
        supports_vision=False
    )
    CODELLAMA = ModelFormat(
        chat_format="llama-2",
        start_sequence="[INST]",
        stop_sequences=["[/INST]", "User:", "</s>"],
        system_template="[SYSTEM]{}[/SYSTEM]",
        supports_vision=False
    )
    TINYLLAMA = ModelFormat(
        chat_format="chatml",
        start_sequence="<|user|>",
        stop_sequences=["<|assistant|>", "<|endoftext|>"],
        system_template="<|system|>{}<|endoftext|>",
        supports_vision=False
    )
    DEFAULT = ModelFormat(
        chat_format="raw",
        start_sequence="",
        stop_sequences=["</s>", "<|endoftext|>"],
        system_template="{}\n\n",
        supports_vision=False
    )

class PromptLibrary:
    # Base mindset - 30 words max
    BASE_MINDSET = """You are Adam, a helpful AI assistant with vision capabilities. Be direct, honest, and practical. Focus on providing clear, useful responses while being friendly and conversational."""

    DEFAULT = PromptTemplate(
        system_prompt=f"""{BASE_MINDSET}

You provide clear, informative, and natural responses. Engage in conversation naturally while providing accurate and relevant information. If you're unsure about something, acknowledge it honestly. Keep responses concise but complete. When providing examples, format them using code blocks like ```html```, ```js```, or ```python``` as appropriate for the content.""",
        temperature=0.7,
        max_tokens=2048,
        stop_sequences=["User:", "\n\n"]
    )

    VISION = PromptTemplate(
        system_prompt=f"""{BASE_MINDSET}

You are a vision-capable AI assistant that can analyze images in detail. When images are provided:
1. Describe what you see comprehensively - objects, people, text, colors, composition, and spatial relationships
2. Answer questions about the visual content with specificity and accuracy
3. Identify text, logos, signs, and other readable elements
4. Analyze artistic elements, emotions, and context when relevant
5. Provide helpful insights about the image content

Be thorough but concise. If asked about something not visible in the image, clearly state that.""",
        temperature=0.7,
        max_tokens=2048,
        stop_sequences=["User:", "\n\n"]
    )

    CODE = PromptTemplate(
        system_prompt=f"""{BASE_MINDSET}

As a programming assistant with vision capabilities, provide clear, well-documented code solutions with explanations. When images of code are provided, analyze them carefully and provide improvements or explanations. Focus on best practices and efficient implementations. When sharing code, use the appropriate language formatting such as ```js``` for JavaScript or ```python``` for Python code.""",
        temperature=0.2,
        max_tokens=2048,
        stop_sequences=["User:", "\n\n"]
    )

    CREATIVE = PromptTemplate(
        system_prompt=f"""{BASE_MINDSET}

As a creative assistant with vision capabilities, generate engaging and imaginative content. When images are provided, use them as inspiration for creative works. Think outside the box while maintaining coherence and quality. When providing examples of creative work, format it using the appropriate language blocks like ```html``` or ```text```.""",
        temperature=0.9,
        max_tokens=2048,
        stop_sequences=["User:", "\n\n"]
    )

class ModelDownloader:
    # Vision-capable models prioritized for GPU setups
    VISION_MODELS = {
        "high_quality": {  # Best quality vision model
            "repo_id": "cjpais/llava-v1.6-mistral-7b-gguf",
            "filename": "llava-v1.6-mistral-7b.Q5_K_M.gguf",
            "expected_size": 5600000000,  # ~5.6GB
            "checksum": None,
            "description": "LLaVA v1.6 with Mistral 7B - High Quality Q5"
        },
        "balanced": {  # Good balance of quality and speed
            "repo_id": "cjpais/llava-v1.6-mistral-7b-gguf", 
            "filename": "llava-v1.6-mistral-7b.Q4_K_M.gguf",
            "expected_size": 4200000000,  # ~4.2GB
            "checksum": None,
            "description": "LLaVA v1.6 with Mistral 7B - Balanced Q4"
        },
        "fast": {  # Faster but lower quality
            "repo_id": "mys/ggml_llava-v1.5-7b",
            "filename": "ggml-model-q4_k.gguf",
            "expected_size": 4000000000,  # ~4GB
            "checksum": None,
            "description": "LLaVA v1.5 - Fast Q4"
        }
    }
    
    # Fallback text-only models if vision fails
    FALLBACK_MODELS = {
        "high_memory": {  # 16GB+ VRAM
            "repo_id": "TheBloke/Llama-2-13B-chat-GGUF",
            "filename": "llama-2-13b-chat.Q4_K_M.gguf",
            "expected_size": 7323000000,
            "description": "Llama 2 13B Chat - Large Model"
        },
        "medium_memory": {  # 8-16GB VRAM  
            "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
            "filename": "llama-2-7b-chat.Q5_K_M.gguf",
            "expected_size": 4781000000,
            "description": "Llama 2 7B Chat - High Quality"
        }
    }

    def __init__(self):
        self.current_dir = Path.cwd()
        self.models_dir = self.current_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        logger.info(f"Models directory: {self.models_dir}")

    def detect_gpu_memory(self) -> str:
        """Detect available GPU memory to choose best model"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                vram_mb = int(result.stdout.strip())
                vram_gb = vram_mb / 1024
                logger.info(f"Detected GPU VRAM: {vram_gb:.1f}GB")
                
                if vram_gb >= 12:  # High-end GPU
                    return "high_quality"
                elif vram_gb >= 8:   # Mid-range GPU  
                    return "balanced"
                else:              # Lower-end GPU
                    return "fast"
            else:
                logger.warning("Could not detect GPU memory, using balanced model")
                return "balanced"
        except Exception as e:
            logger.warning(f"GPU memory detection failed: {e}, using balanced model")
            return "balanced"

    def get_optimal_vision_model(self) -> Dict:
        """Get the best vision model for current hardware"""
        gpu_category = self.detect_gpu_memory()
        optimal_model = self.VISION_MODELS[gpu_category]
        logger.info(f"Selected {gpu_category} vision model: {optimal_model['filename']}")
        logger.info(f"Model description: {optimal_model['description']}")
        return optimal_model

    def find_existing_gguf(self) -> Optional[str]:
        """Find existing GGUF files with preference for vision models"""
        search_patterns = [
            str(self.current_dir / "*.gguf"),
            str(self.models_dir / "*.gguf")
        ]
        
        for pattern in search_patterns:
            gguf_files = glob.glob(pattern)
            if gguf_files:
                logger.info(f"Found existing GGUF files: {gguf_files}")
                
                # Strongly prefer vision models
                vision_files = [f for f in gguf_files if any(keyword in f.lower() 
                               for keyword in ['llava', 'vision', 'multimodal', 'clip'])]
                
                if vision_files:
                    # Sort vision models by quality (Q5 > Q4 > Q3) and size
                    def vision_model_score(filename):
                        name = filename.lower()
                        score = 0
                        if 'q5' in name: score += 50
                        elif 'q4' in name: score += 40  
                        elif 'q3' in name: score += 30
                        
                        if 'v1.6' in name: score += 20
                        elif 'v1.5' in name: score += 10
                        
                        if 'mistral' in name: score += 5
                        
                        return score
                    
                    vision_files.sort(key=vision_model_score, reverse=True)
                    selected_file = vision_files[0]
                    logger.info(f"Selected best vision model: {selected_file}")
                    return selected_file
                
                # If no vision models, select best text model
                def model_quality_score(filename):
                    name = filename.lower()
                    if 'q5' in name: return 5
                    elif 'q4' in name: return 4
                    elif 'q3' in name: return 3
                    else: return 1
                
                gguf_files.sort(key=lambda x: (model_quality_score(x), os.path.getsize(x)), reverse=True)
                selected_file = gguf_files[0]
                logger.info(f"Selected best existing model: {selected_file}")
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
            logger.info(f"Total file size: {total_size / (1024*1024*1024):.2f} GB")
            
            downloaded = 0
            chunk_size = 8192 * 16  # Larger chunks for faster download
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress every 500MB
                        if downloaded % (500 * 1024 * 1024) == 0 or downloaded >= total_size:
                            progress = (downloaded / total_size * 100) if total_size > 0 else 0
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded / (1024*1024*1024):.2f} GB)")
            
            logger.info(f"Download completed: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    def download_model(self, model_info: Dict) -> Optional[str]:
        """Download a specific model"""
        model_path = self.models_dir / model_info["filename"]
        
        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            return str(model_path)
        
        logger.info(f"Downloading vision model: {model_info['filename']}...")
        logger.info(f"Description: {model_info['description']}")
        
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
        
        return None

    def get_model_path(self) -> Optional[str]:
        """Get model path - prioritize vision models"""
        # First check for existing models (with vision preference)
        existing_model = self.find_existing_gguf()
        if existing_model:
            return existing_model
        
        # Download optimal vision model for current hardware
        logger.info("No existing model found, downloading optimal vision model...")
        optimal_model_info = self.get_optimal_vision_model()
        
        vision_model_path = self.download_model(optimal_model_info)
        if vision_model_path:
            return vision_model_path
        
        # Fallback to text-only model if vision download fails
        logger.warning("Vision model download failed, falling back to text-only model")
        gpu_memory = "medium_memory" if self.detect_gpu_memory() == "balanced" else "high_memory"
        fallback_info = self.FALLBACK_MODELS[gpu_memory]
        return self.download_model(fallback_info)

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

class ImageProcessor:
    def __init__(self):
        self.temp_files = []

    def cleanup_temp_files(self):
        """Clean up temporary image files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {str(e)}")
        self.temp_files = []

    def process_base64_images(self, images_data: List[Dict]) -> List[str]:
        """Process base64 images and return file paths"""
        image_paths = []
        
        if not PIL_AVAILABLE:
            logger.error("PIL not available for image processing")
            return image_paths
        
        for idx, image_data in enumerate(images_data):
            try:
                # Extract base64 data
                if isinstance(image_data, dict):
                    base64_data = image_data.get('data', image_data.get('imageBase64', ''))
                    image_name = image_data.get('name', f'image_{idx}')
                else:
                    base64_data = image_data
                    image_name = f'image_{idx}'
                
                # Clean base64 string
                if 'base64,' in base64_data:
                    base64_data = base64_data.split('base64,')[1]
                
                # Decode image
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_bytes))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.jpg', delete=False, prefix=f'img_{idx}_'
                )
                self.temp_files.append(temp_file.name)
                
                # Save image
                image.save(temp_file.name, 'JPEG', quality=95)
                image_paths.append(temp_file.name)
                
                logger.info(f"Processed image {idx}: {image.size} pixels -> {temp_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing image {idx}: {str(e)}")
                continue
        
        return image_paths

class EnhancedResponseGenerator:
    def __init__(self):
        self.llm: Optional[Any] = None
        self.model_format: Optional[ModelFormat] = None
        self.model_path: Optional[str] = None
        self.model_config: Dict[str, Any] = {
            "n_ctx": MAX_CONTEXT_SIZE,
            "n_threads": 4,  # Increased for better performance
            "verbose": False,
            "repeat_penalty": 1.1,  # Reduced for better coherence
            "top_p": 0.95,
            "top_k": 40,
            "n_batch": 512,  # Increased for better GPU utilization
            "use_mlock": False,
            "use_mmap": True
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else None
        self.current_model_type = "gguf"
        self.downloader = ModelDownloader()
        self.image_processor = ImageProcessor()
        self.supports_vision = False
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
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return {"available": False, "error": str(e)}

    def calculate_safe_max_tokens(self, input_text: str, max_context: int = None) -> int:
        """Calculate safe max_tokens based on input length"""
        if max_context is None:
            max_context = self.model_config["n_ctx"]
        
        estimated_input_tokens = len(input_text) // 4
        system_overhead = 400  # Increased for vision models
        available_tokens = max_context - estimated_input_tokens - system_overhead
        safe_tokens = max(256, min(available_tokens, 2048))  # Increased max
        
        logger.info(f"Input chars: {len(input_text)}, Estimated tokens: {estimated_input_tokens}, Safe max_tokens: {safe_tokens}")
        
        return safe_tokens

    def detect_model_format(self, model_path: str) -> ModelFormat:
        """Detect model format from model path"""
        model_name = os.path.basename(model_path).lower()
        logger.info(f"Detecting format for model: {model_name}")
        
        if "llava" in model_name:
            if "v1.6" in model_name or "mistral" in model_name:
                logger.info("Detected LLaVA v1.6/Mistral vision model")
                self.supports_vision = True
                return ModelFormats.LLAVA_V16
            else:
                logger.info("Detected LLaVA v1.5 vision model")
                self.supports_vision = True
                return ModelFormats.LLAVA
        elif any(keyword in model_name for keyword in ["vision", "multimodal", "clip"]):
            logger.info("Detected vision model")
            self.supports_vision = True
            return ModelFormats.LLAVA
        elif "codellama" in model_name:
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
            logger.info(f"Using default Llama2 format for model: {model_name}")
            return ModelFormats.LLAMA2

    def initialize_model(self):
        """Initialize the model - download if necessary and load"""
        try:
            logger.info("Initializing model...")
            
            if not LLAMA_CPP_AVAILABLE:
                logger.error("llama-cpp-python not installed. Please install with: pip install llama-cpp-python")
                raise ImportError("llama-cpp-python not installed")
            
            self.model_path = self.downloader.get_model_path()
            if not self.model_path:
                raise FileNotFoundError("Could not find or download a GGUF model")
            
            logger.info(f"Using model: {self.model_path}")
            self.model_format = self.detect_model_format(self.model_path)
            self.load_model()
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def load_model(self):
        """Load the GGUF model with vision support"""
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading GGUF model: {self.model_path}")
            
            config = self.model_config.copy()
            config["chat_format"] = self.model_format.chat_format
            
            # Enhanced settings for vision models
            if self.supports_vision:
                config.update({
                    "n_ctx": min(4096, MAX_CONTEXT_SIZE),  # Vision models need more context
                    "clip_model_path": None,  # Auto-detect CLIP model
                    "logits_all": True,  # Required for some vision models
                })
                logger.info("Configured for vision model with CLIP support")
            
            # Try GPU first, fallback to CPU
            try:
                logger.info("Attempting to load model with GPU acceleration...")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=-1,  # Use all GPU layers
                    **config
                )
                logger.info("Model loaded successfully with GPU acceleration")
            except Exception as gpu_error:
                logger.warning(f"GPU loading failed: {str(gpu_error)}. Falling back to CPU.")
                config["n_gpu_layers"] = 0
                self.llm = Llama(
                    model_path=self.model_path,
                    **config
                )
                logger.info("Model loaded successfully on CPU")
            
            logger.info(f"Model loaded with chat format: {self.model_format.chat_format}")
            logger.info(f"Vision support: {self.supports_vision}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_prompt_template(self, response_type: str, has_images: bool = False) -> PromptTemplate:
        """Get the appropriate prompt template based on response type and image presence"""
        if has_images and self.supports_vision:
            return PromptLibrary.VISION
        
        type_mapping = {
            ResponseType.CHAT.value: PromptLibrary.DEFAULT,
            ResponseType.ANALYSIS.value: PromptLibrary.DEFAULT,
            ResponseType.CREATIVE.value: PromptLibrary.CREATIVE,
            ResponseType.CODE.value: PromptLibrary.CODE,
            ResponseType.VISION.value: PromptLibrary.VISION
        }
        return type_mapping.get(response_type, PromptLibrary.DEFAULT)

    def prepare_chat_messages(self, template: PromptTemplate, user_input: str, response_type: str, image_paths: List[str] = None) -> List[Dict[str, Any]]:
        """Prepare chat messages with proper formatting for text and images"""
        messages = [
            {"role": "system", "content": template.system_prompt}
        ]
        
        # Prepare user message
        if image_paths and self.supports_vision:
            # For vision models, include images in the message
            user_content = [{"type": "text", "text": user_input}]
            
            for image_path in image_paths:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_path}"}
                })
            
            messages.append({"role": "user", "content": user_content})
            logger.info(f"Prepared vision message with {len(image_paths)} images")
        else:
            # Text-only message
            if image_paths:
                # Add image information to text for non-vision models
                image_info = f"\n\n[Note: {len(image_paths)} image(s) were provided but this model doesn't support vision. Please describe what you'd like me to help you with regarding these images.]"
                user_input += image_info
            
            messages.append({"role": "user", "content": user_input})
        
        return messages

    def get_generation_params(self, template: PromptTemplate, response_type: str, input_text: str = "") -> Dict[str, Any]:
        """Get generation parameters based on response type and input length"""
        safe_max_tokens = self.calculate_safe_max_tokens(input_text)
        
        params = {
            "max_tokens": min(template.max_tokens, safe_max_tokens),
            "temperature": template.temperature,
            "top_p": self.model_config["top_p"],
            "repeat_penalty": self.model_config["repeat_penalty"],
            "stop": self.model_format.stop_sequences if self.model_format else template.stop_sequences
        }

        # Adjust parameters by response type
        if response_type == "code":
            params["temperature"] = 0.1  # More deterministic for code
        elif response_type == "creative":
            params["temperature"] = 0.9  # More creative
        elif response_type == "vision":
            params["temperature"] = 0.7  # Balanced for vision tasks

        memory_info = self.check_memory_usage()
        if memory_info.get("available") and memory_info.get("percent", 0) > 80:
            params["max_tokens"] = min(params["max_tokens"], 1024)
            logger.warning(f"High memory usage ({memory_info['percent']:.1f}%), reducing max_tokens to {params['max_tokens']}")

        return params

    def generate_response(self, user_input: str, response_type: str = "chat", images_data: List[Dict] = None) -> str:
        """Generate response using the loaded model with vision support"""
        if not user_input and not images_data:
            return json.dumps({
                "conversation_response": "Please provide some input or images.",
                "status": "error"
            })

        try:
            if self.llm is None:
                raise ValueError("Model not initialized")

            # Process images if provided
            image_paths = []
            if images_data:
                logger.info(f"Processing {len(images_data)} images for vision analysis")
                image_paths = self.image_processor.process_base64_images(images_data)
                logger.info(f"Successfully processed {len(image_paths)} images")
                
                # If images provided but no text, set default vision prompt
                if not user_input:
                    user_input = "Please analyze these images and describe what you see in detail."
                    response_type = "vision"

            memory_info = self.check_memory_usage()
            if memory_info.get("available") and memory_info.get("percent", 0) > 85:
                logger.warning(f"High memory usage: {memory_info}")

            logger.info(f"Generating response for type: {response_type}, with {len(image_paths)} images")
            template = self.get_prompt_template(response_type, bool(image_paths))
            messages = self.prepare_chat_messages(template, user_input, response_type, image_paths)
            generation_params = self.get_generation_params(template, response_type, user_input)

            logger.debug(f"Generation parameters: {generation_params}")

            try:
                output = self.llm.create_chat_completion(
                    messages=messages,
                    **generation_params
                )
            except Exception as gen_error:
                if "context" in str(gen_error).lower() or "length" in str(gen_error).lower():
                    generation_params["max_tokens"] = min(generation_params["max_tokens"], 512)
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
                        "supports_vision": self.supports_vision,
                        "images_processed": len(image_paths),
                        "persona": "Adam",
                        "input_length": len(user_input),
                        "vision_capable": self.supports_vision
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
                "conversation_response": "Request too large. Please reduce input size or image count.",
                "status": "error",
                "error_type": "memory_limit"
            })
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            
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
        finally:
            # Clean up temporary image files
            self.image_processor.cleanup_temp_files()

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
@rate_limit(max_requests=30, window_seconds=60)
def get_response():
    """Generate response endpoint with enhanced image support"""
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
        
        # Handle multiple image formats
        images_data = []
        
        # Check for multiple images
        if 'images' in data and data['images']:
            images_data = data['images']
        elif 'imageBase64s' in data and data['imageBase64s']:
            images_data = data['imageBase64s']
        elif 'imageBase64' in data and data['imageBase64']:
            # Single image for backward compatibility
            images_data = [data['imageBase64']]

        if not user_input and not images_data:
            return jsonify({"error": "No text or images provided"}), 400

        if is_base64 and user_input:
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

        response_type = data.get('response_type', 'vision' if images_data else 'chat')

        if response_type not in [rt.value for rt in ResponseType]:
            return jsonify({
                "error": f"Invalid response_type. Must be one of: {[rt.value for rt in ResponseType]}"
            }), 400

        logger.info(f"Processing request - Type: {response_type}, Input length: {len(user_input)}, Images: {len(images_data)}")
        
        # Check memory before processing
        memory_info = generator.check_memory_usage()
        if memory_info.get("available") and memory_info.get("percent", 0) > 90:
            return jsonify({
                "error": "Server memory usage too high. Please try again later.",
                "status": "error",
                "error_type": "memory_limit"
            }), 503

        # Warn if images provided but vision not supported
        if images_data and not generator.supports_vision:
            logger.warning(f"Images provided but model doesn't support vision. Model: {generator.model_path}")

        response = generator.generate_response(user_input, response_type, images_data)
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
        "supports_vision": generator.supports_vision if generator else False,
        "vision_enabled": ENABLE_VISION,
        "persona": "Adam",
        "memory_info": memory_info,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_context_size": MAX_CONTEXT_SIZE,
        "image_processing": PIL_AVAILABLE
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
        "persona": "Adam - Vision-Capable AI Assistant",
        "mindset": PromptLibrary.BASE_MINDSET,
        "model_info": {
            "path": generator.model_path,
            "format": generator.model_format.chat_format if generator.model_format else None,
            "loaded": generator.llm is not None,
            "supports_vision": generator.supports_vision,
            "context_size": generator.model_config["n_ctx"]
        },
        "capabilities": {
            "text_generation": True,
            "code_generation": True,
            "creative_writing": True,
            "image_analysis": generator.supports_vision,
            "multi_image_support": generator.supports_vision and PIL_AVAILABLE
        },
        "server_info": {
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "hf_hub_available": HF_HUB_AVAILABLE,
            "psutil_available": PSUTIL_AVAILABLE,
            "pil_available": PIL_AVAILABLE,
            "vision_enabled": ENABLE_VISION
        },
        "limits": {
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "max_context_size": MAX_CONTEXT_SIZE
        },
        "memory_info": memory_info
    })

@app.route('/test_vision', methods=['POST'])
def test_vision():
    """Test endpoint specifically for vision capabilities"""
    if not generator or not generator.supports_vision:
        return jsonify({
            "error": "Vision capabilities not available",
            "supports_vision": False
        }), 400
    
    return jsonify({
        "message": "Vision capabilities are active",
        "supports_vision": True,
        "model_format": generator.model_format.chat_format,
        "instructions": "Send images via 'images' array in POST request to /generate_response"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting Adam AI server with vision capabilities on port {port}")
    logger.info(f"Max prompt length: {MAX_PROMPT_LENGTH}")
    logger.info(f"Max context size: {MAX_CONTEXT_SIZE}")
    logger.info(f"Vision support enabled: {ENABLE_VISION}")
    logger.info(f"PIL available for image processing: {PIL_AVAILABLE}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
