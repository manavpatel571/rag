"""
PDF Processing Module
Converts PDF to markdown using docling and extracts images
"""
import os
import io
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions


class PDFProcessor:
    def __init__(self, output_dir: str = "extracted_images", use_gpu: bool = True):
        """Initialize PDF processor with GPU support"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure pipeline for GPU acceleration
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = {
            "device": "cuda" if use_gpu else "cpu"
        }
        
        # Initialize converter with GPU-enabled pipeline
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfPipelineOptions(
                    accelerator_options={"device": "cuda" if use_gpu else "cpu"}
                )
            }
        )
        
        device_name = "GPU (CUDA)" if use_gpu else "CPU"
        print(f"✅ Docling initialized with {device_name} acceleration")
    
    def extract_images_from_pdf(self, pdf_path: str, pdf_name: str) -> List[Dict]:
        """Extract images from PDF and save them"""
        images_data = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Create image from bytes
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Save image
                    image_filename = f"{pdf_name}_page_{page_num + 1}_img_{img_index + 1}.png"
                    image_path = os.path.join(self.output_dir, image_filename)
                    image.save(image_path)
                    
                    images_data.append({
                        "path": image_path,
                        "page": page_num + 1,
                        "index": img_index + 1,
                        "filename": image_filename
                    })
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error extracting images: {e}")
        
        return images_data
    
    def convert_to_markdown(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Convert PDF to markdown and extract metadata"""
        try:
            # Convert PDF to markdown using docling
            result = self.converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()
            
            # Get document structure for page information
            pages_content = []
            
            # Parse the result to get page-wise content
            # Handle both old and new docling API
            if hasattr(result.document, 'pages'):
                for page_idx, page in enumerate(result.document.pages):
                    page_text = ""
                    # Check if page has elements attribute
                    if hasattr(page, 'elements'):
                        for element in page.elements:
                            if hasattr(element, 'export_to_markdown'):
                                page_text += element.export_to_markdown() + "\n"
                            else:
                                page_text += str(element) + "\n"
                    else:
                        # Fallback: get text from page
                        page_text = str(page)
                    
                    pages_content.append({
                        "page": page_idx + 1,
                        "content": page_text.strip()
                    })
            else:
                # If pages not available, split markdown by page markers
                pages = markdown_content.split('## Page ')
                for idx, page_content in enumerate(pages[1:], 1):
                    pages_content.append({
                        "page": idx,
                        "content": page_content.strip()
                    })
            
            return markdown_content, pages_content
            
        except Exception as e:
            print(f"Error converting PDF to markdown: {e}")
            # Fallback to basic text extraction
            return self._fallback_text_extraction(pdf_path)
    
    def _fallback_text_extraction(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Fallback method using PyMuPDF for text extraction"""
        try:
            pdf_document = fitz.open(pdf_path)
            full_text = ""
            pages_content = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                full_text += f"\n## Page {page_num + 1}\n\n{page_text}\n"
                
                pages_content.append({
                    "page": page_num + 1,
                    "content": page_text.strip()
                })
            
            pdf_document.close()
            return full_text, pages_content
            
        except Exception as e:
            print(f"Error in fallback extraction: {e}")
            return "", []
    
    def enrich_with_image_descriptions(self, pages_content: List[Dict], 
                                       markdown_content: str, 
                                       described_images: List[Dict]) -> Tuple[List[Dict], str]:
        """
        Enrich page content and markdown with image descriptions.
        Inserts image descriptions into their respective pages.
        
        Args:
            pages_content: List of page dictionaries with 'page' and 'content' keys
            markdown_content: Full markdown content
            described_images: List of image dictionaries with 'page', 'description', 'filename' keys
        
        Returns:
            Tuple of (enriched_pages_content, enriched_markdown)
        """
        # Group images by page
        page_images = {}
        for img in described_images:
            page_num = img.get("page", 0)
            if page_num not in page_images:
                page_images[page_num] = []
            page_images[page_num].append(img)
        
        # Enrich pages_content
        enriched_pages = []
        for page_data in pages_content:
            page_num = page_data["page"]
            page_text = page_data["content"]
            
            # Add image descriptions to this page
            if page_num in page_images:
                page_text += "\n\n---\n**[IMAGES ON THIS PAGE]:**\n"
                for img in page_images[page_num]:
                    img_desc = img.get("description", "No description")
                    img_filename = img.get("filename", "unknown")
                    page_text += f"\n- **Image: {img_filename}**\n  {img_desc}\n"
            
            enriched_pages.append({
                "page": page_num,
                "content": page_text
            })
        
        # Enrich markdown content by inserting descriptions after each page
        enriched_markdown = markdown_content
        for page_num in sorted(page_images.keys(), reverse=True):  # Reverse to avoid position shifts
            images = page_images[page_num]
            
            # Create image section for this page
            image_section = "\n\n---\n**[IMAGES ON THIS PAGE]:**\n"
            for img in images:
                img_desc = img.get("description", "No description")
                img_filename = img.get("filename", "unknown")
                image_section += f"\n- **Image: {img_filename}**\n  {img_desc}\n"
            
            # Find the position to insert (after page marker)
            page_marker = f"## Page {page_num}"
            if page_marker in enriched_markdown:
                # Find next page marker or end of document
                next_page_marker = f"## Page {page_num + 1}"
                
                if next_page_marker in enriched_markdown:
                    # Insert before next page
                    enriched_markdown = enriched_markdown.replace(
                        next_page_marker, 
                        image_section + "\n\n" + next_page_marker
                    )
                else:
                    # Last page - append at the end
                    enriched_markdown += image_section
        
        return enriched_pages, enriched_markdown
    
    def process_pdf(self, pdf_path: str, pdf_name: str = None) -> Dict:
        """Main method to process PDF: extract text and images"""
        if pdf_name is None:
            pdf_name = Path(pdf_path).stem
        
        # Convert to markdown
        markdown_content, pages_content = self.convert_to_markdown(pdf_path)
        
        # Extract images
        images_data = self.extract_images_from_pdf(pdf_path, pdf_name)
        
        # Map images to pages
        page_image_map = {}
        for img_data in images_data:
            page = img_data["page"]
            if page not in page_image_map:
                page_image_map[page] = []
            page_image_map[page].append(img_data)
        
        return {
            "markdown": markdown_content,
            "pages": pages_content,
            "images": images_data,
            "page_image_map": page_image_map,
            "pdf_name": pdf_name
        }

