"""
PDF Cache Module
Manages caching of processed PDFs to avoid reprocessing
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional


class PDFCache:
    """Cache manager for processed PDFs"""
    
    def __init__(self, cache_dir: str = "pdf_cache"):
        """Initialize PDF cache"""
        self.cache_dir = cache_dir
        self.markdown_dir = os.path.join(cache_dir, "markdown")
        self.metadata_dir = os.path.join(cache_dir, "metadata")
        
        # Create directories
        os.makedirs(self.markdown_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def _get_pdf_hash(self, pdf_path: str) -> str:
        """Generate hash for PDF file"""
        hash_md5 = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_cache_key(self, pdf_name: str, pdf_hash: str) -> str:
        """Generate cache key from PDF name and hash"""
        return f"{pdf_name}_{pdf_hash[:8]}"
    
    def is_cached(self, pdf_path: str, pdf_name: str) -> bool:
        """Check if PDF is already cached"""
        try:
            pdf_hash = self._get_pdf_hash(pdf_path)
            cache_key = self._get_cache_key(pdf_name, pdf_hash)
            
            metadata_file = os.path.join(self.metadata_dir, f"{cache_key}.json")
            return os.path.exists(metadata_file)
        except Exception as e:
            print(f"Error checking cache: {e}")
            return False
    
    def get_cached_data(self, pdf_path: str, pdf_name: str) -> Optional[Dict]:
        """Get cached processing data for a PDF"""
        try:
            pdf_hash = self._get_pdf_hash(pdf_path)
            cache_key = self._get_cache_key(pdf_name, pdf_hash)
            
            metadata_file = os.path.join(self.metadata_dir, f"{cache_key}.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
        except Exception as e:
            print(f"Error reading cache: {e}")
            return None
    
    def save_to_cache(self, pdf_path: str, pdf_name: str, processed_data: Dict, markdown_content: str):
        """Save processed PDF data to cache"""
        try:
            pdf_hash = self._get_pdf_hash(pdf_path)
            cache_key = self._get_cache_key(pdf_name, pdf_hash)
            
            # Save markdown
            markdown_file = os.path.join(self.markdown_dir, f"{cache_key}.md")
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # Prepare metadata (without circular references)
            cache_metadata = {
                "pdf_name": pdf_name,
                "pdf_hash": pdf_hash,
                "cache_key": cache_key,
                "num_pages": len(processed_data.get('pages', [])),
                "num_images": len(processed_data.get('images', [])),
                "markdown_file": markdown_file,
                "images": processed_data.get('images', []),
                "page_image_map": {str(k): v for k, v in processed_data.get('page_image_map', {}).items()}
            }
            
            # Save metadata
            metadata_file = os.path.join(self.metadata_dir, f"{cache_key}.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(cache_metadata, f, indent=2)
            
            print(f"✅ Cached PDF data: {cache_key}")
            
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def get_markdown(self, pdf_path: str, pdf_name: str) -> Optional[str]:
        """Get cached markdown for a PDF"""
        try:
            pdf_hash = self._get_pdf_hash(pdf_path)
            cache_key = self._get_cache_key(pdf_name, pdf_hash)
            
            markdown_file = os.path.join(self.markdown_dir, f"{cache_key}.md")
            
            if os.path.exists(markdown_file):
                with open(markdown_file, 'r', encoding='utf-8') as f:
                    return f.read()
            
            return None
        except Exception as e:
            print(f"Error reading markdown: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data"""
        import shutil
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.markdown_dir, exist_ok=True)
                os.makedirs(self.metadata_dir, exist_ok=True)
                print("🧹 Cache cleared")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached PDFs"""
        try:
            metadata_files = list(Path(self.metadata_dir).glob("*.json"))
            
            cached_pdfs = []
            for meta_file in metadata_files:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached_pdfs.append({
                        "name": data.get("pdf_name"),
                        "cache_key": data.get("cache_key"),
                        "pages": data.get("num_pages"),
                        "images": data.get("num_images")
                    })
            
            return {
                "total_cached": len(cached_pdfs),
                "pdfs": cached_pdfs
            }
        except Exception as e:
            print(f"Error getting cache info: {e}")
            return {"total_cached": 0, "pdfs": []}

