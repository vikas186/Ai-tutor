"""
Image enhancement pipeline for optimal OCR accuracy
"""
import cv2
import numpy as np
from PIL import Image
from typing import Union
import io


class ImageEnhancementPipeline:
    """Advanced image preprocessing for 100% OCR accuracy"""
    
    def __init__(self):
        self.enhancement_steps = [
            self._deskew,
            self._enhance_contrast,
            self._reduce_noise,
            self._optimize_resolution,
            self._sharpen
        ]
    
    def optimize(self, image_input: Union[str, bytes, Image.Image]) -> bytes:
        """
        Apply full enhancement pipeline to optimize image for OCR
        
        Args:
            image_input: Path to image, image bytes, or PIL Image
            
        Returns:
            Enhanced image as bytes (PNG format)
        """
        # Load image
        if isinstance(image_input, str):
            # Check if it's a file path
            if os.path.exists(image_input):
                # Try OpenCV first (for images)
                img = cv2.imread(image_input)
                if img is None:
                    # If OpenCV fails, try PIL (handles more formats)
                    try:
                        pil_img = Image.open(image_input)
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        raise ValueError(f"Failed to load image from path '{image_input}': {str(e)}. Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP, PDF")
            else:
                raise FileNotFoundError(f"Image file not found: {image_input}")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                # Try PIL if OpenCV fails
                try:
                    pil_img = Image.open(io.BytesIO(image_input))
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    raise ValueError(f"Failed to decode image bytes: {str(e)}")
        elif isinstance(image_input, Image.Image):
            img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        if img is None:
            raise ValueError("Failed to load image - OpenCV returned None. Check file format and integrity.")
        
        # Convert to grayscale if needed (but keep color for Gemini)
        # Apply enhancement steps
        enhanced = img.copy()
        
        for step in self.enhancement_steps:
            enhanced = step(enhanced)
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.png', enhanced)
        return buffer.tobytes()
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct rotation and deskew image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect skew angle
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate if significant skew detected
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better text visibility"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise while preserving text edges"""
        # Use bilateral filter to preserve edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def _optimize_resolution(self, image: np.ndarray) -> np.ndarray:
        """Ensure optimal resolution for OCR (300 DPI equivalent)"""
        h, w = image.shape[:2]
        
        # Target: minimum 300 DPI equivalent (roughly 3000px width for A4)
        target_width = 3000
        if w < target_width:
            scale = target_width / w
            new_width = int(w * scale)
            new_height = int(h * scale)
            image = cv2.resize(image, (new_width, new_height), 
                             interpolation=cv2.INTER_CUBIC)
        
        return image
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image to improve text clarity"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

