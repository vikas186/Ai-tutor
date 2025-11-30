"""
File handler for supporting multiple file formats (PDF, images)
"""
import os
from pathlib import Path
from typing import Union, List
import logging
from PIL import Image
import io

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("pdf2image not available. PDF support disabled.")

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle different file formats and convert to images"""
    
    @staticmethod
    def is_pdf(file_path: str) -> bool:
        """Check if file is a PDF"""
        return Path(file_path).suffix.lower() == '.pdf'
    
    @staticmethod
    def is_image(file_path: str) -> bool:
        """Check if file is an image"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        return Path(file_path).suffix.lower() in image_extensions
    
    @staticmethod
    def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images (optional - only if poppler is available)
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion (default 300)
            
        Returns:
            List of PIL Images (one per page)
        """
        if not PDF_SUPPORT:
            raise ValueError("PDF to image conversion not available. Install pdf2image and poppler. Note: Gemini API supports PDFs directly, so conversion is optional.")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            logger.info(f"Converted PDF to {len(images)} page(s)")
            return images
        except Exception as e:
            error_msg = str(e)
            if "poppler" in error_msg.lower() or "PATH" in error_msg:
                raise ValueError(f"Poppler not installed or not in PATH. For PDF support, either: 1) Install poppler, or 2) Use Gemini's native PDF support (recommended). Error: {error_msg}")
            logger.error(f"Failed to convert PDF: {error_msg}")
            raise ValueError(f"Failed to convert PDF to images: {error_msg}")
    
    @staticmethod
    def convert_pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF bytes to list of PIL Images
        
        Args:
            pdf_bytes: PDF file as bytes
            dpi: Resolution for conversion (default 300)
            
        Returns:
            List of PIL Images (one per page)
        """
        if not PDF_SUPPORT:
            raise ValueError("PDF support not available. Install pdf2image: pip install pdf2image")
        
        try:
            images = convert_from_bytes(pdf_bytes, dpi=dpi)
            logger.info(f"Converted PDF bytes to {len(images)} page(s)")
            return images
        except Exception as e:
            logger.error(f"Failed to convert PDF bytes: {str(e)}")
            raise ValueError(f"Failed to convert PDF bytes to images: {str(e)}")
    
    @staticmethod
    def get_first_page_image(file_path: str) -> Image.Image:
        """
        Get first page/image from file (PDF or image)
        
        Args:
            file_path: Path to file (PDF or image)
            
        Returns:
            PIL Image of first page
        """
        if FileHandler.is_pdf(file_path):
            images = FileHandler.convert_pdf_to_images(file_path)
            if not images:
                raise ValueError("PDF has no pages")
            return images[0]  # Return first page
        elif FileHandler.is_image(file_path):
            try:
                return Image.open(file_path)
            except Exception as e:
                raise ValueError(f"Failed to open image: {str(e)}")
        else:
            raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")
    
    @staticmethod
    def validate_file(file_path: str) -> dict:
        """
        Validate file and return information
        
        Returns:
            dict with file info: {'type': 'pdf'|'image'|'unknown', 'exists': bool, 'size': int}
        """
        result = {
            'exists': os.path.exists(file_path),
            'size': 0,
            'type': 'unknown',
            'extension': Path(file_path).suffix.lower()
        }
        
        if result['exists']:
            result['size'] = os.path.getsize(file_path)
            
            if FileHandler.is_pdf(file_path):
                result['type'] = 'pdf'
            elif FileHandler.is_image(file_path):
                result['type'] = 'image'
        
        return result

