"""
Configuration file for PDF RAG system
All models run via HuggingFace API, Docling can use GPU
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None


class Config:
    """System configuration"""
    
    # GPU Settings
    USE_GPU = GPU_AVAILABLE
    GPU_NAME = GPU_NAME
    
    # API Keys
    HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    
    # Model Settings (via HuggingFace API)
    CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    IMAGE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local embeddings
    
    # Processing Settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 5
    
    # Directories
    EXTRACTED_IMAGES_DIR = "extracted_images"
    CHROMA_DB_DIR = "chroma_db"
    
    @staticmethod
    def get_api_status():
        """Get API configuration status"""
        return {
            "hf_api_configured": Config.HF_API_KEY is not None and Config.HF_API_KEY != "",
            "chat_model": Config.CHAT_MODEL,
            "image_model": Config.IMAGE_MODEL,
            "embedding_model": Config.EMBEDDING_MODEL,
            "gpu_available": Config.USE_GPU,
            "gpu_name": Config.GPU_NAME
        }
    
    @staticmethod
    def print_config():
        """Print current configuration"""
        status = Config.get_api_status()
        print("=" * 60)
        print("PDF RAG System Configuration")
        print("=" * 60)
        print(f"API: HuggingFace Router")
        print(f"API Configured: {'✅ Yes' if status['hf_api_configured'] else '❌ No'}")
        print(f"GPU Available: {'✅ Yes' if status['gpu_available'] else '❌ No'}")
        if status['gpu_name']:
            print(f"GPU: {status['gpu_name']}")
        print(f"Chat Model: {Config.CHAT_MODEL}")
        print(f"Image Model: {Config.IMAGE_MODEL}")
        print(f"Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"Chunk Size: {Config.CHUNK_SIZE}")
        print(f"Top K Results: {Config.TOP_K_RESULTS}")
        print("=" * 60)

