"""
Image Description Module
Uses Qwen2.5-VL-7B-Instruct via HuggingFace API for advanced image understanding
"""
import os
import base64
from typing import Dict, List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()


class ImageDescriber:
    def __init__(self):
        """Initialize image describer with Qwen2.5-VL-7B-Instruct via HuggingFace API"""
        self.api_key = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
        
        if not self.api_key:
            print("⚠️ Warning: HF_API_KEY not found in environment")
            self.client = None
        else:
            # Initialize InferenceClient with HuggingFace
            self.client = InferenceClient(
                provider="auto",
                api_key=self.api_key
            )
            print("✅ Qwen2.5-VL-7B initialized with HuggingFace API")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            return image_data
    
    def describe_image(self, image_path: str, prompt: str = None) -> str:
        """
        Generate description for an image using Qwen2.5-VL-7B-Instruct
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt for description (default: detailed description)
        """
        if self.client is None:
            return "[Image from document - API key not configured]"
        
        try:
            # Encode image to base64
            image_base64 = self.encode_image_to_base64(image_path)
            
            # Determine image format
            ext = image_path.lower().split('.')[-1]
            mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else "image/png"
            
            # Default prompt for detailed description
            if prompt is None:
                prompt = "Describe this image in detail, including all visible elements, text, charts, diagrams, and their relationships."
            
            # Create chat completion with image
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.3
            )
            
            # Extract message content
            description = response.choices[0].message["content"]
            return description.strip()
            
        except Exception as e:
            print(f"Error describing image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return f"[Image from document - error: {str(e)}]"
    
    def describe_images_batch(self, images_data: List[Dict]) -> List[Dict]:
        """
        Process a batch of images and generate descriptions
        
        Args:
            images_data: List of dicts with image metadata (path, filename, page, etc.)
        
        Returns:
            List of dicts with added 'description' field
        """
        described_images = []
        total = len(images_data)
        
        for idx, img_data in enumerate(images_data, 1):
            image_path = img_data["path"]
            print(f"📸 Processing image {idx}/{total}: {img_data['filename']}")
            
            # Generate description
            description = self.describe_image(image_path)
            
            # Add description to metadata
            described_images.append({
                **img_data,
                "description": description
            })
        
        return described_images
    
    def get_image_caption(self, image_path: str, brief: bool = False) -> str:
        """
        Get a brief or detailed caption for an image
        
        Args:
            image_path: Path to the image
            brief: If True, request a one-sentence description
        """
        if brief:
            prompt = "Describe this image in one sentence."
        else:
            prompt = None  # Use default detailed prompt
        
        return self.describe_image(image_path, prompt=prompt)
