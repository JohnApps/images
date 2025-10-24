# image_analyzer.py
# V3
"""
        self.conn.execute("set memory_limit = '12GB'")
        self.conn.execute("set max_memory = '12GB'")
"""
# image_analyzer_ai.py
"""
Image analyzer with AI vision integration for real object detection.
Supports multiple vision backends: transformers (BLIP/CLIP), torch vision, or cloud APIs.
"""

import os
import duckdb
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
import numpy as np
from collections import Counter
import hashlib

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# AI Vision imports - will try multiple options
AI_BACKEND = None
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel
    import torch
    AI_BACKEND = "transformers"
    print("✓ Using Transformers (BLIP/CLIP) for AI vision")
except ImportError:
    print("⚠ transformers not available, trying torchvision...")
    try:
        import torchvision.models as models
        import torchvision.transforms as transforms
        import torch
        AI_BACKEND = "torchvision"
        print("✓ Using TorchVision for AI vision")
    except ImportError:
        print("⚠ No AI vision library available - will use basic analysis only")
        print("Install with: pip install transformers torch torchvision")


class AIVisionAnalyzer:
    """AI-powered vision analyzer using transformer models"""
    
    def __init__(self):
        """Initialize AI vision models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_model = None
        self.blip_processor = None
        self.clip_model = None
        self.clip_processor = None
        
        if AI_BACKEND == "transformers":
            self._load_transformers()
        elif AI_BACKEND == "torchvision":
            self._load_torchvision()
    
    def _load_transformers(self):
        """Load BLIP and CLIP models"""
        try:
            print("Loading BLIP model for image captioning...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            
            print("Loading CLIP model for classification...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            print("✓ AI models loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading models: {e}")
    
    def _load_torchvision(self):
        """Load torchvision ResNet model"""
        try:
            print("Loading ResNet model...")
            self.resnet_model = models.resnet50(pretrained=True)
            self.resnet_model.eval()
            self.resnet_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print("✓ ResNet model loaded")
        except Exception as e:
            print(f"⚠ Error loading ResNet: {e}")
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate natural language caption for image"""
        if not self.blip_model:
            return "AI caption unavailable - model not loaded"
        
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Error generating caption: {e}"
    
    def classify_scene(self, image: Image.Image) -> List[Tuple[str, float]]:
        """Classify image content using CLIP"""
        if not self.clip_model:
            return []
        
        try:
            # Define categories to check
            categories = [
                "a photo of a butterfly on a flower",
                "a photo of an insect",
                "a photo of a flower",
                "a photo of nature",
                "a photo of wildlife",
                "a photo of a garden",
                "a photo of a bee",
                "a photo of a moth",
                "a photo of a dandelion",
                "a photo of a sunflower",
                "a close-up macro photo",
                "a landscape photo",
                "a portrait photo",
                "a photo of vegetation",
                "a photo of grass and plants"
            ]
            
            inputs = self.clip_processor(
                text=categories, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get top 5 predictions
            top5_prob, top5_idx = torch.topk(probs[0], 5)
            results = [
                (categories[idx.item()].replace("a photo of ", ""), prob.item())
                for idx, prob in zip(top5_idx, top5_prob)
            ]
            
            return results
        except Exception as e:
            print(f"Error in classification: {e}")
            return []
    
    def detect_objects(self, image: Image.Image) -> Dict:
        """Detect and describe objects in image"""
        results = {
            'caption': None,
            'scene_classifications': [],
            'confidence_scores': {}
        }
        
        if AI_BACKEND == "transformers":
            # Generate caption
            results['caption'] = self.generate_caption(image)
            
            # Classify scene
            classifications = self.classify_scene(image)
            results['scene_classifications'] = classifications
            
            if classifications:
                results['confidence_scores'] = {
                    cat: float(score) for cat, score in classifications
                }
        
        return results


class ImageAnalyzer:
    """Enhanced image analyzer with AI vision capabilities"""
    
    def __init__(self, db_path: str = "images.db", use_ai: bool = True):
        """Initialize analyzer with optional AI vision"""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.conn.execute("set memory_limit = '12GB'")
        self.conn.execute("set max_memory = '12GB'")
        self._init_database()
        self.batch_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # Initialize AI vision
        self.use_ai = use_ai and AI_BACKEND is not None
        self.ai_analyzer = None
        
        if self.use_ai:
            print("\nInitializing AI Vision...")
            self.ai_analyzer = AIVisionAnalyzer()
        else:
            print("\n⚠ Running without AI vision - interpretations will be limited")
    
    def _init_database(self):
        """Initialize database with AI results table"""
        
        # Previous tables remain the same...
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id VARCHAR PRIMARY KEY,
                batch_id VARCHAR,
                file_path VARCHAR,
                file_name VARCHAR,
                file_size_bytes BIGINT,
                file_hash VARCHAR,
                created_date TIMESTAMP,
                modified_date TIMESTAMP,
                analysis_date TIMESTAMP,
                width INTEGER,
                height INTEGER,
                aspect_ratio DOUBLE,
                file_format VARCHAR,
                color_mode VARCHAR,
                has_exif BOOLEAN
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS camera_info (
                image_id VARCHAR PRIMARY KEY,
                make VARCHAR,
                model VARCHAR,
                lens_model VARCHAR,
                focal_length DOUBLE,
                focal_length_35mm DOUBLE,
                aperture DOUBLE,
                shutter_speed VARCHAR,
                iso INTEGER,
                flash VARCHAR,
                exposure_mode VARCHAR,
                white_balance VARCHAR,
                metering_mode VARCHAR
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS content_analysis (
                image_id VARCHAR PRIMARY KEY,
                dominant_colors JSON,
                color_palette JSON,
                brightness_avg DOUBLE,
                brightness_std DOUBLE,
                contrast_score DOUBLE,
                sharpness_estimate DOUBLE,
                saturation_avg DOUBLE,
                red_channel_avg DOUBLE,
                green_channel_avg DOUBLE,
                blue_channel_avg DOUBLE,
                color_distribution JSON,
                image_histogram JSON
            )
        """)
        
        # New AI analysis table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_analysis (
                image_id VARCHAR PRIMARY KEY,
                ai_caption VARCHAR,
                detected_objects JSON,
                scene_classifications JSON,
                confidence_scores JSON,
                ai_backend VARCHAR
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                image_id VARCHAR PRIMARY KEY,
                accuracy_rating INTEGER,
                feedback_timestamp TIMESTAMP,
                interpretation_text VARCHAR
            )
        """)
        
        self.conn.commit()
        print(f"Database initialized: {self.db_path}")
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_exif_data(self, image: Image.Image) -> Dict:
        """Extract EXIF data"""
        exif_data = {}
        try:
            exif = image._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    try:
                        exif_data[tag] = str(value)
                    except:
                        exif_data[tag] = "<<unable to convert>>"
        except:
            pass
        return exif_data
    
    def _parse_camera_info(self, exif_data: Dict) -> Dict:
        """Parse camera settings"""
        camera_info = {
            'make': exif_data.get('Make', None),
            'model': exif_data.get('Model', None),
            'lens_model': exif_data.get('LensModel', None),
            'focal_length': None,
            'focal_length_35mm': None,
            'aperture': None,
            'shutter_speed': exif_data.get('ExposureTime', None),
            'iso': None,
            'flash': exif_data.get('Flash', None),
            'exposure_mode': exif_data.get('ExposureMode', None),
            'white_balance': exif_data.get('WhiteBalance', None),
            'metering_mode': exif_data.get('MeteringMode', None)
        }
        
        # Parse numeric values
        focal_length_str = exif_data.get('FocalLength', '')
        if focal_length_str and '/' in str(focal_length_str):
            try:
                num, denom = focal_length_str.split('/')
                camera_info['focal_length'] = float(num) / float(denom)
            except:
                pass
        
        fnumber = exif_data.get('FNumber', '')
        if fnumber and '/' in str(fnumber):
            try:
                num, denom = fnumber.split('/')
                camera_info['aperture'] = float(num) / float(denom)
            except:
                pass
        
        iso = exif_data.get('ISOSpeedRatings', None)
        if iso:
            try:
                camera_info['iso'] = int(iso)
            except:
                pass
        
        return camera_info
    
    def _analyze_image_content(self, image: Image.Image) -> Dict:
        """Basic statistical analysis"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        analysis = {
            'brightness_avg': float(np.mean(img_array)),
            'brightness_std': float(np.std(img_array)),
            'red_channel_avg': float(np.mean(img_array[:, :, 0])),
            'green_channel_avg': float(np.mean(img_array[:, :, 1])),
            'blue_channel_avg': float(np.mean(img_array[:, :, 2]))
        }
        
        gray = np.mean(img_array, axis=2)
        analysis['contrast_score'] = float(np.std(gray))
        
        # Saturation
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b).astype(float)
        min_rgb = np.minimum(np.minimum(r, g), b).astype(float)
        
        saturation = np.zeros_like(max_rgb, dtype=float)
        mask = max_rgb > 0
        saturation[mask] = (max_rgb[mask] - min_rgb[mask]) / max_rgb[mask]
        analysis['saturation_avg'] = float(np.mean(saturation))
        
        # Sharpness
        try:
            from scipy import ndimage
            laplacian = ndimage.laplace(gray)
            analysis['sharpness_estimate'] = float(np.var(laplacian))
        except:
            analysis['sharpness_estimate'] = None
        
        # Dominant colors
        pixels = img_array.reshape(-1, 3)
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        quantized = (pixels // 32) * 32
        unique_colors = [tuple(int(c) for c in color) for color in quantized]
        color_counts = Counter(unique_colors)
        dominant_colors = [
            {'color': [int(c) for c in color], 'count': int(count)}
            for color, count in color_counts.most_common(10)
        ]
        analysis['dominant_colors'] = json.dumps(dominant_colors)
        
        # Color palette
        color_bins = {
            'red': 0, 'green': 0, 'blue': 0,
            'yellow': 0, 'cyan': 0, 'magenta': 0,
            'white': 0, 'black': 0, 'gray': 0
        }
        
        for pixel in pixels:
            r_val, g_val, b_val = int(pixel[0]), int(pixel[1]), int(pixel[2])
            total = r_val + g_val + b_val
            
            if total < 50:
                color_bins['black'] += 1
            elif total > 700:
                color_bins['white'] += 1
            elif max(r_val, g_val, b_val) - min(r_val, g_val, b_val) < 50:
                color_bins['gray'] += 1
            elif r_val > g_val and r_val > b_val:
                if g_val > 150:
                    color_bins['yellow'] += 1
                else:
                    color_bins['red'] += 1
            elif g_val > r_val and g_val > b_val:
                if b_val > 150:
                    color_bins['cyan'] += 1
                else:
                    color_bins['green'] += 1
            else:
                if r_val > 150:
                    color_bins['magenta'] += 1
                else:
                    color_bins['blue'] += 1
        
        analysis['color_palette'] = json.dumps(color_bins)
        
        # Histogram
        hist_r, _ = np.histogram(r, bins=16, range=(0, 256))
        hist_g, _ = np.histogram(g, bins=16, range=(0, 256))
        hist_b, _ = np.histogram(b, bins=16, range=(0, 256))
        
        analysis['image_histogram'] = json.dumps({
            'red': [int(x) for x in hist_r],
            'green': [int(x) for x in hist_g],
            'blue': [int(x) for x in hist_b]
        })
        
        # Color distribution
        analysis['color_distribution'] = json.dumps({
            'mean_red': float(np.mean(r)),
            'mean_green': float(np.mean(g)),
            'mean_blue': float(np.mean(b)),
            'std_red': float(np.std(r)),
            'std_green': float(np.std(g)),
            'std_blue': float(np.std(b))
        })
        
        return analysis
    
    def _generate_interpretation(self, content_analysis: Dict, camera_info: Dict,
                                ai_results: Dict, width: int, height: int) -> str:
        """Generate interpretation with AI insights"""
        
        lines = []
        
        # START WITH AI CAPTION if available
        if ai_results and ai_results.get('caption'):
            lines.append(f"AI Description: {ai_results['caption']}")
            lines.append("")
        
        # AI Scene Classifications
        if ai_results and ai_results.get('scene_classifications'):
            top_classes = ai_results['scene_classifications'][:3]
            class_str = ", ".join([f"{cat} ({score:.1%})" for cat, score in top_classes])
            lines.append(f"Detected: {class_str}")
            lines.append("")
        
        # Composition
        aspect = width / height if height > 0 else 1
        if aspect > 1.3:
            orientation = "landscape"
        elif aspect < 0.8:
            orientation = "portrait"
        else:
            orientation = "square"
        lines.append(f"Format: {width}x{height} {orientation}")
        
        # Technical analysis
        brightness = content_analysis.get('brightness_avg', 0)
        if brightness < 85:
            b_desc = "dark/underexposed"
        elif brightness < 140:
            b_desc = "well-exposed"
        else:
            b_desc = "bright"
        lines.append(f"Exposure: {b_desc} ({brightness:.1f}/255)")
        
        contrast = content_analysis.get('contrast_score', 0)
        lines.append(f"Contrast: {contrast:.1f}")
        
        saturation = content_analysis.get('saturation_avg', 0)
        lines.append(f"Saturation: {saturation:.2f}")
        
        sharpness = content_analysis.get('sharpness_estimate')
        if sharpness:
            lines.append(f"Sharpness: {sharpness:.1f}")
        
        # Camera info
        if camera_info and camera_info.get('make'):
            lines.append(f"\nCamera: {camera_info['make']} {camera_info['model']}")
            settings = []
            if camera_info.get('focal_length'):
                settings.append(f"{camera_info['focal_length']:.0f}mm")
            if camera_info.get('aperture'):
                settings.append(f"f/{camera_info['aperture']:.1f}")
            if camera_info.get('iso'):
                settings.append(f"ISO{camera_info['iso']}")
            if settings:
                lines.append(f"Settings: {' | '.join(settings)}")
        
        return "\n".join(lines)
    
    def _display_image_with_interpretation(self, image_path: str, interpretation: str) -> Optional[int]:
        """Display and request feedback"""
        try:
            import matplotlib.pyplot as plt
            
            image = Image.open(image_path)
            fig = plt.figure(figsize=(16, 10))
            
            ax_img = plt.subplot(1, 2, 1)
            ax_img.imshow(image)
            ax_img.axis('off')
            ax_img.set_title(os.path.basename(image_path), fontsize=14, fontweight='bold')
            
            ax_text = plt.subplot(1, 2, 2)
            ax_text.axis('off')
            
            ax_text.text(0.05, 0.95, 'IMAGE INTERPRETATION', 
                        transform=ax_text.transAxes,
                        fontsize=16, fontweight='bold', 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax_text.text(0.05, 0.85, interpretation,
                        transform=ax_text.transAxes,
                        fontsize=11, verticalalignment='top',
                        family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax_text.text(0.05, 0.05, 
                        'Please review and provide accuracy rating in terminal...',
                        transform=ax_text.transAxes,
                        fontsize=10, style='italic',
                        verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)
            
            print("\n" + "="*60)
            print("ACCURACY RATING")
            print("="*60)
            print("Rate interpretation accuracy (10-90%):")
            print("  10-30 = Poor")
            print("  31-50 = Fair")
            print("  51-70 = Good")
            print("  71-90 = Excellent")
            print("Type 'skip' to skip")
            print("-"*60)
            
            while True:
                try:
                    user_input = input("Rating (10-90) or 'skip': ").strip().lower()
                    
                    if user_input in ['skip', 'q', 's']:
                        plt.close(fig)
                        return None
                    
                    rating = int(user_input)
                    
                    if 10 <= rating <= 90:
                        print(f"✓ Rating: {rating}%")
                        plt.close(fig)
                        return rating
                    else:
                        print("⚠ Must be 10-90")
                
                except ValueError:
                    print("⚠ Enter number 10-90 or 'skip'")
                except KeyboardInterrupt:
                    plt.close(fig)
                    return None
        
        except Exception as e:
            print(f"⚠ Display error: {e}")
            return None
    
    def analyze_image(self, file_path: str, request_feedback: bool = True) -> Optional[str]:
        """Analyze image with AI vision"""
        try:
            image_id = str(uuid.uuid4())
            
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            file_hash = self._compute_file_hash(file_path)
            created_date = datetime.fromtimestamp(file_stat.st_ctime)
            modified_date = datetime.fromtimestamp(file_stat.st_mtime)
            
            image = Image.open(file_path)
            width, height = image.size
            aspect_ratio = width / height if height > 0 else 0
            
            print(f"  Analyzing: {os.path.basename(file_path)}")
            print(f"    Size: {width}x{height}")
            
            # Extract EXIF
            exif_data = self._extract_exif_data(image)
            has_exif = len(exif_data) > 0
            camera_info = self._parse_camera_info(exif_data) if exif_data else {}
            
            # Basic analysis
            content_analysis = self._analyze_image_content(image)
            
            # AI ANALYSIS
            ai_results = {}
            if self.use_ai and self.ai_analyzer:
                print("    Running AI vision analysis...")
                ai_results = self.ai_analyzer.detect_objects(image)
                
                if ai_results.get('caption'):
                    print(f"    AI: {ai_results['caption']}")
                
                if ai_results.get('scene_classifications'):
                    top = ai_results['scene_classifications'][0]
                    print(f"    Detected: {top[0]} ({top[1]:.1%})")
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                content_analysis, camera_info, ai_results, width, height
            )
            
            # Store in database
            self.conn.execute("""
                INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                image_id, self.batch_id, file_path, os.path.basename(file_path),
                file_size, file_hash, created_date, modified_date, datetime.now(),
                width, height, aspect_ratio, image.format, image.mode, has_exif
            ])
            
            if camera_info:
                self.conn.execute("""
                    INSERT INTO camera_info VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    image_id, camera_info['make'], camera_info['model'],
                    camera_info['lens_model'], camera_info['focal_length'],
                    camera_info['focal_length_35mm'], camera_info['aperture'],
                    camera_info['shutter_speed'], camera_info['iso'],
                    camera_info['flash'], camera_info['exposure_mode'],
                    camera_info['white_balance'], camera_info['metering_mode']
                ])
            
            self.conn.execute("""
                INSERT INTO content_analysis VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                image_id,
                content_analysis.get('dominant_colors'),
                content_analysis.get('color_palette'),
                content_analysis.get('brightness_avg'),
                content_analysis.get('brightness_std'),
                content_analysis.get('contrast_score'),
                content_analysis.get('sharpness_estimate'),
                content_analysis.get('saturation_avg'),
                content_analysis.get('red_channel_avg'),
                content_analysis.get('green_channel_avg'),
                content_analysis.get('blue_channel_avg'),
                content_analysis.get('color_distribution'),
                content_analysis.get('image_histogram')
            ])
            
            # Store AI results
            if ai_results:
                self.conn.execute("""
                    INSERT INTO ai_analysis VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    image_id,
                    ai_results.get('caption'),
                    json.dumps(ai_results.get('detected_objects', [])),
                    json.dumps(ai_results.get('scene_classifications', [])),
                    json.dumps(ai_results.get('confidence_scores', {})),
                    AI_BACKEND
                ])
            
            self.conn.commit()
            image.close()
            
            # Request feedback
            if request_feedback:
                rating = self._display_image_with_interpretation(file_path, interpretation)
                if rating:
                    self.conn.execute("""
                        INSERT INTO user_feedback VALUES (?, ?, ?, ?)
                    """, [image_id, rating, datetime.now(), interpretation])
                    self.conn.commit()
            
            return image_id
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def close(self):
        """Close database"""
        if hasattr(self, 'conn'):
            self.conn.close()


def main():
    """Main function"""
    IMAGE_DIRECTORY = r"O:\Bilder\1-D7100\2015-06-30-Lauenen"
    
    print("IMAGE ANALYZER WITH AI VISION")
    print("="*60)
    
    analyzer = ImageAnalyzer(db_path="images.db", use_ai=True)
    
    # Process single image or directory
    from pathlib import Path
    image_files = list(Path(IMAGE_DIRECTORY).glob("*.jpg"))[:5]  # First 5 for testing
    
    for img_file in image_files:
        analyzer.analyze_image(str(img_file), request_feedback=True)
        print()
    
    analyzer.close()
    print("✓ Complete!")


if __name__ == "__main__":
    main()