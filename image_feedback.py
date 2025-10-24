# image_feedback.py
# image_feedback.py
"""
Simple image feedback system - displays single image with interpretation
and collects accuracy rating (10-90%) from terminal.
"""

import os
import duckdb
import json
from datetime import datetime
from typing import Optional, Dict
import uuid
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt


class ImageFeedbackCollector:
    """Display image with interpretation and collect user feedback"""
    
    def __init__(self, db_path: str = "images.db"):
        """Initialize with database connection"""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_feedback_table()
    
    def _init_feedback_table(self):
        """Initialize feedback table in database"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS image_feedback (
                feedback_id VARCHAR PRIMARY KEY,
                image_path VARCHAR,
                image_name VARCHAR,
                width INTEGER,
                height INTEGER,
                interpretation_text VARCHAR,
                accuracy_rating INTEGER,
                feedback_timestamp TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def _extract_basic_exif(self, image: Image.Image) -> Dict:
        """Extract basic EXIF data"""
        exif_data = {}
        try:
            exif = image._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)
        except:
            pass
        return exif_data
    
    def _parse_camera_settings(self, exif_data: Dict) -> Dict:
        """Parse camera settings from EXIF"""
        camera = {
            'make': exif_data.get('Make', 'Unknown'),
            'model': exif_data.get('Model', 'Unknown'),
            'focal_length': None,
            'aperture': None,
            'iso': None,
            'shutter_speed': exif_data.get('ExposureTime', None)
        }
        
        # Parse focal length
        fl = exif_data.get('FocalLength', '')
        if fl and '/' in str(fl):
            try:
                num, denom = fl.split('/')
                camera['focal_length'] = float(num) / float(denom)
            except:
                pass
        
        # Parse aperture
        fn = exif_data.get('FNumber', '')
        if fn and '/' in str(fn):
            try:
                num, denom = fn.split('/')
                camera['aperture'] = float(num) / float(denom)
            except:
                pass
        
        # Parse ISO
        iso = exif_data.get('ISOSpeedRatings', None)
        if iso:
            try:
                camera['iso'] = int(iso)
            except:
                pass
        
        return camera
    
    def _analyze_content(self, image: Image.Image) -> Dict:
        """Analyze image content"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        analysis = {
            'brightness_avg': float(np.mean(img_array)),
            'contrast_score': float(np.std(np.mean(img_array, axis=2))),
            'saturation_avg': 0.0,
            'red_avg': float(np.mean(img_array[:, :, 0])),
            'green_avg': float(np.mean(img_array[:, :, 1])),
            'blue_avg': float(np.mean(img_array[:, :, 2]))
        }
        
        # Calculate saturation (with proper handling of edge cases)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b).astype(float)
        min_rgb = np.minimum(np.minimum(r, g), b).astype(float)
        
        # Use np.divide with where parameter to handle division by zero
        saturation = np.zeros_like(max_rgb, dtype=float)
        mask = max_rgb > 0
        saturation[mask] = (max_rgb[mask] - min_rgb[mask]) / max_rgb[mask]
        
        analysis['saturation_avg'] = float(np.mean(saturation))
        
        return analysis
    
    def _generate_interpretation(self, analysis: Dict, camera: Dict, 
                                 width: int, height: int) -> str:
        """Generate interpretation text"""
        lines = []
        
        # Composition
        aspect = width / height if height > 0 else 1
        if aspect > 1.3:
            orientation = "landscape"
        elif aspect < 0.8:
            orientation = "portrait"
        else:
            orientation = "square"
        lines.append(f"Format: {width}x{height} {orientation}")
        
        # Brightness
        brightness = analysis['brightness_avg']
        if brightness < 85:
            b_desc = "dark/underexposed"
        elif brightness < 140:
            b_desc = "well-exposed"
        else:
            b_desc = "bright/overexposed"
        lines.append(f"Exposure: {b_desc} ({brightness:.1f}/255)")
        
        # Contrast
        contrast = analysis['contrast_score']
        if contrast < 35:
            c_desc = "low contrast, flat"
        elif contrast < 60:
            c_desc = "moderate contrast"
        else:
            c_desc = "high contrast, dramatic"
        lines.append(f"Contrast: {c_desc} ({contrast:.1f})")
        
        # Saturation
        saturation = analysis['saturation_avg']
        if saturation < 0.2:
            s_desc = "muted/desaturated"
        elif saturation < 0.4:
            s_desc = "natural saturation"
        else:
            s_desc = "vivid/saturated"
        lines.append(f"Saturation: {s_desc} ({saturation:.2f})")
        
        # Color cast
        r_avg = analysis['red_avg']
        g_avg = analysis['green_avg']
        b_avg = analysis['blue_avg']
        
        if r_avg > g_avg + 20 and r_avg > b_avg + 20:
            cast = "warm/reddish"
        elif b_avg > r_avg + 20 and b_avg > g_avg + 20:
            cast = "cool/bluish"
        elif g_avg > r_avg + 15:
            cast = "greenish (vegetation?)"
        else:
            cast = "neutral"
        lines.append(f"Color: {cast} cast")
        
        # Camera
        if camera['make'] != 'Unknown':
            lines.append(f"\nCamera: {camera['make']} {camera['model']}")
            settings = []
            if camera['focal_length']:
                settings.append(f"{camera['focal_length']:.0f}mm")
            if camera['aperture']:
                settings.append(f"f/{camera['aperture']:.1f}")
            if camera['iso']:
                settings.append(f"ISO{camera['iso']}")
            if camera['shutter_speed']:
                settings.append(f"{camera['shutter_speed']}s")
            if settings:
                lines.append(f"Settings: {' | '.join(settings)}")
        
        # Scene interpretation
        lines.append("\nScene Analysis:")
        if saturation < 0.25 and brightness < 100:
            lines.append("- Indoor/low-light scene")
        elif saturation > 0.35 and brightness > 140:
            lines.append("- Outdoor daylight scene")
        
        if g_avg > (r_avg + b_avg) / 2 + 15:
            lines.append("- Vegetation/foliage present")
        
        if b_avg > (r_avg + g_avg) / 2 + 15:
            lines.append("- Sky/water elements likely")
        
        if camera['focal_length']:
            if camera['focal_length'] < 28:
                lines.append("- Wide-angle perspective")
            elif camera['focal_length'] > 70:
                lines.append("- Telephoto perspective")
        
        return "\n".join(lines)
    
    def collect_feedback(self, image_path: str) -> Optional[int]:
        """
        Display image with interpretation and collect feedback.
        Returns accuracy rating (10-90) or None if skipped.
        """
        if not os.path.exists(image_path):
            print(f"ERROR: Image not found: {image_path}")
            return None
        
        try:
            # Load and analyze image
            image = Image.open(image_path)
            width, height = image.size
            
            print(f"\nAnalyzing: {os.path.basename(image_path)}")
            print(f"Size: {width}x{height}")
            
            # Extract EXIF and analyze
            exif_data = self._extract_basic_exif(image)
            camera = self._parse_camera_settings(exif_data)
            analysis = self._analyze_content(image)
            
            # Generate interpretation
            interpretation = self._generate_interpretation(analysis, camera, width, height)
            
            # Display image with interpretation
            fig = plt.figure(figsize=(16, 10))
            
            # Image on left
            ax_img = plt.subplot(1, 2, 1)
            ax_img.imshow(image)
            ax_img.axis('off')
            ax_img.set_title(os.path.basename(image_path), 
                           fontsize=14, fontweight='bold', pad=20)
            
            # Interpretation on right
            ax_text = plt.subplot(1, 2, 2)
            ax_text.axis('off')
            
            ax_text.text(0.05, 0.95, 'IMAGE INTERPRETATION', 
                        transform=ax_text.transAxes,
                        fontsize=16, fontweight='bold', 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax_text.text(0.05, 0.85, interpretation,
                        transform=ax_text.transAxes,
                        fontsize=12, verticalalignment='top',
                        family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax_text.text(0.05, 0.05, 
                        'Review and provide accuracy rating in terminal...',
                        transform=ax_text.transAxes,
                        fontsize=10, style='italic',
                        verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)
            
            # Get user rating
            print("\n" + "="*60)
            print("ACCURACY RATING")
            print("="*60)
            print("Rate the interpretation accuracy (10-90%):")
            print("  10-30 = Poor")
            print("  31-50 = Fair")
            print("  51-70 = Good")
            print("  71-90 = Excellent")
            print("Type 'skip' to skip this image")
            print("-"*60)
            
            while True:
                try:
                    user_input = input("Rating (10-90) or 'skip': ").strip().lower()
                    
                    if user_input in ['skip', 'q', 's']:
                        print("Skipped.")
                        plt.close(fig)
                        return None
                    
                    rating = int(user_input)
                    
                    if 10 <= rating <= 90:
                        # Store in database
                        feedback_id = str(uuid.uuid4())
                        self.conn.execute("""
                            INSERT INTO image_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            feedback_id,
                            image_path,
                            os.path.basename(image_path),
                            width,
                            height,
                            interpretation,
                            rating,
                            datetime.now()
                        ])
                        self.conn.commit()
                        
                        print(f"✓ Rating saved: {rating}%")
                        plt.close(fig)
                        return rating
                    else:
                        print("⚠ Rating must be 10-90")
                
                except ValueError:
                    print("⚠ Enter a number 10-90 or 'skip'")
                except KeyboardInterrupt:
                    print("\n⚠ Interrupted")
                    plt.close(fig)
                    return None
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def view_feedback_history(self, limit: int = 10):
        """Display recent feedback history"""
        print("\n" + "="*60)
        print("FEEDBACK HISTORY")
        print("="*60)
        
        results = self.conn.execute(f"""
            SELECT 
                image_name,
                accuracy_rating,
                feedback_timestamp
            FROM image_feedback
            ORDER BY feedback_timestamp DESC
            LIMIT {limit}
        """).fetchall()
        
        if results:
            for img_name, rating, timestamp in results:
                print(f"{timestamp}: {img_name} - {rating}%")
            
            # Statistics
            stats = self.conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(accuracy_rating) as avg_rating,
                    MIN(accuracy_rating) as min_rating,
                    MAX(accuracy_rating) as max_rating
                FROM image_feedback
            """).fetchone()
            
            print("\nStatistics:")
            print(f"  Total ratings: {stats[0]}")
            print(f"  Average: {stats[1]:.1f}%")
            print(f"  Range: {stats[2]}% - {stats[3]}%")
        else:
            print("No feedback recorded yet")
        
        print("="*60)
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()


def main():
    """Main function for single image feedback"""
    
    print("IMAGE INTERPRETATION FEEDBACK SYSTEM")
    print("="*60)
    print("This program displays one image at a time with interpretation")
    print("and requests accuracy rating (10-90%) from terminal.")
    print("="*60)
    
    # Initialize collector
    collector = ImageFeedbackCollector(db_path="images.db")
    
    # Get image path from user
    print("\nEnter image path (or 'quit' to exit):")
    image_path = input("> ").strip()
    
    if image_path.lower() in ['quit', 'q', 'exit']:
        print("Exiting...")
        collector.close()
        return
    
    # Remove quotes if present
    image_path = image_path.strip('"').strip("'")
    
    # Collect feedback
    rating = collector.collect_feedback(image_path)
    
    if rating:
        print(f"\n✓ Feedback recorded: {rating}%")
    
    # Show history
    collector.view_feedback_history()
    
    # Close
    collector.close()
    
    print(f"\n✓ Data stored in: images.db")


if __name__ == "__main__":
    main()