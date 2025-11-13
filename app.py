# ===============================
# Film Roll Photo Processor - OpenCV 16-bit Version
# ===============================

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
import base64
import os
import sys
import json
import time
from datetime import datetime
import rawpy
from PIL import Image, ImageFile
from white_balance import auto_white_balance as auto_white_balance_new

# Force stdout to flush immediately for Docker logs
sys.stdout.reconfigure(line_buffering=True)

# -----------------------------
# Flask Setup
# -----------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 750 * 1024 * 1024 * 1024
app.config['OUTPUT_BASE'] = os.environ.get('OUTPUT_FOLDER', './output')
Path(app.config['OUTPUT_BASE']).mkdir(exist_ok=True)

image_cache = {}

progress_data = {
    'current': 0,
    'total': 0,
    'current_file': '',
    'status': '',
    'start_time': 0
}

DEFAULT_SETTINGS = {
    'passthrough_mode': False,
    'auto_white_balance': False,
    'auto_rotate': True,
    'auto_straighten': True,
    'remove_borders': True,
    'white_balance_strength': 1.0,
    'white_balance_method': 'gray_world'  or 'white_patch',
    'preserve_luminance': True,
    'border_threshold_low': 10,
    'border_threshold_high': 245,
    'straighten_threshold': 0.5,
    'jpeg_quality': 95,
    'border_buffer_percent': 5,
    'use_tiff_compression': False
}

current_settings = DEFAULT_SETTINGS.copy()

# -----------------------------
# Helper Functions
# -----------------------------

def extract_exif_from_raw(file_bytes):
    """Extract EXIF metadata from RAW file using multiple methods"""
    metadata = {}
    
    # Method 1: Try exifread (best for RAW files)
    try:
        import exifread
        import io
        
        tags = exifread.process_file(io.BytesIO(file_bytes), details=False)
        
        # Convert exifread tags to standard format
        for tag, value in tags.items():
            if tag.startswith('EXIF') or tag.startswith('Image'):
                # Clean up tag name
                clean_tag = tag.split()[-1]
                
                # Convert value to string or number
                if hasattr(value, 'values'):
                    metadata[clean_tag] = str(value)
                else:
                    metadata[clean_tag] = str(value)
        
        if metadata:
            print(f"  EXIF extracted with exifread: {len(metadata)} tags")
            return metadata
            
    except ImportError:
        print(f"  exifread not available, trying PIL...")
    except Exception as e:
        print(f"  exifread failed: {e}")
    
    # Method 2: Try PIL (works for some RAW formats with embedded JPEG)
    try:
        import io
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        img = Image.open(io.BytesIO(file_bytes))
        exif_data = img.getexif()
        
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                # Convert to string if needed
                if isinstance(value, bytes):
                    try:
                        metadata[tag] = value.decode('utf-8', errors='ignore')
                    except:
                        metadata[tag] = str(value)
                else:
                    metadata[tag] = value
            
            print(f"  EXIF extracted with PIL: {len(metadata)} tags")
            return metadata
            
    except Exception as e:
        print(f"  PIL EXIF extraction failed: {e}")
    
    # Method 3: Try rawpy metadata
    try:
        import io
        import rawpy
        
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            # Get camera info from rawpy
            metadata['Make'] = 'Unknown'
            metadata['Model'] = 'Unknown'
            
            # rawpy doesn't expose EXIF directly well, but we can try
            print(f"  Basic metadata from rawpy")
            
    except Exception as e:
        print(f"  rawpy metadata extraction failed: {e}")
    
    return metadata


def load_raw_image(file_bytes):
    """Load RAW image using rawpy with 16-bit output"""
    try:
        import io
        
        # Extract EXIF first (before rawpy consumes the bytes)
        metadata = extract_exif_from_raw(file_bytes)
        
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            # Load with 16-bit and linear gamma
            rgb = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                no_auto_bright=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=16,
                gamma=(1, 1),  # Linear gamma
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
            )
            print(f"  RAW loaded: shape={rgb.shape}, dtype={rgb.dtype}, range=[{rgb.min()}, {rgb.max()}]")
            if metadata:
                print(f"  EXIF extracted: {len(metadata)} tags")
            
            # rawpy outputs RGB, OpenCV uses BGR
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr, metadata
    except Exception as e:
        print(f"Error loading RAW image: {e}")
        return None, {}


def load_tiff_safe(file_bytes):
    """Load TIFF with Pillow, convert to OpenCV BGR format"""
    try:
        import io
        from PIL import Image, ImageFile
        from PIL.ExifTags import TAGS
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(io.BytesIO(file_bytes))
        
        # Extract EXIF
        metadata = {}
        try:
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    metadata[tag] = value
        except:
            pass
        
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array (RGB)
        rgb_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        return bgr_array, metadata
    except Exception as e:
        print(f"Error loading TIFF with Pillow: {e}")
        return None, {}


def remove_borders(img, threshold_low=10, threshold_high=245, buffer_percent=1):
    """Crop borders with proper 16-bit support"""
    is_16bit = img.dtype == np.uint16
    
    # Scale thresholds for 16-bit
    if is_16bit:
        threshold_low = int(threshold_low * 257)  # 257 = 65535/255
        threshold_high = int(threshold_high * 257)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find content area (not too dark, not too bright)
    _, thresh_low = cv2.threshold(gray, threshold_low, 65535 if is_16bit else 255, cv2.THRESH_BINARY)
    _, thresh_high = cv2.threshold(gray, threshold_high, 65535 if is_16bit else 255, cv2.THRESH_BINARY_INV)
    
    # Combine masks
    combined = cv2.bitwise_and(thresh_low, thresh_high)
    
    # Convert to 8-bit for findContours
    if is_16bit:
        combined = (combined / 256).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Apply buffer (crop inward)
        h_buf = int(h * (buffer_percent / 100))
        w_buf = int(w * (buffer_percent / 100))
        
        y0 = max(0, y + h_buf)
        x0 = max(0, x + w_buf)
        y1 = min(img.shape[0], y + h - h_buf)
        x1 = min(img.shape[1], x + w - w_buf)
        
        print(f"  Border crop: {img.shape[:2]} -> ({y0}:{y1}, {x0}:{x1}) = ({y1-y0}x{x1-x0})")
        
        if y1 > y0 and x1 > x0 and (y1 - y0) > 100 and (x1 - x0) > 100:
            return img[y0:y1, x0:x1]
    
    return img


def detect_rotation(img):
    """Detect rotation angle - conservative for scanner alignment"""
    is_16bit = img.dtype == np.uint16
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to 8-bit for edge detection
    if is_16bit:
        gray = (gray / 256).astype(np.uint8)
    
    # Threshold to find borders
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Find edges
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    # Hough lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None or len(lines) < 4:
        print(f"  No rotation detected (insufficient lines)")
        return 0
    
    angles = []
    for rho, theta in lines[:20, 0]:
        angle = np.degrees(theta) - 90
        
        # Normalize to -45 to 45
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        # Only accept small angles (scanner misalignment is typically < 5°)
        if abs(angle) < 5:
            angles.append(angle)
    
    if len(angles) < 4:
        print(f"  No rotation detected (no consistent angles)")
        return 0
    
    median_angle = np.median(angles)
    std_angle = np.std(angles)
    
    print(f"  Rotation: {median_angle:.2f}° ±{std_angle:.2f}° (from {len(angles)} lines)")
    
    # Only rotate if consistent and significant
    if abs(median_angle) > 0.3 and abs(median_angle) < 5 and std_angle < 2:
        return -median_angle
    
    print(f"  Rotation rejected (too large, inconsistent, or too small)")
    return 0


def rotate_image(img, angle):
    """Rotate image preserving 16-bit"""
    is_16bit = img.dtype == np.uint16
    border_val = 65535 if is_16bit else 255
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust matrix
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    
    print(f"  Rotating {angle:.2f}°: {img.shape[:2]} -> ({new_h}, {new_w})")
    
    return cv2.warpAffine(img, matrix, (new_w, new_h), 
                          borderMode=cv2.BORDER_CONSTANT, 
                          borderValue=(border_val, border_val, border_val))

def auto_white_balance(img, strength=1.0, method=None):
    """Apply auto white balance with proper bit depth handling"""
    if method is None:
        method = current_settings.get('white_balance_method', 'gray_world')
    preserve_lum = current_settings.get('preserve_luminance', True)
    return auto_white_balance_new(
        img, 
        strength=strength, 
        method=method,
        preserve_luminance=preserve_lum
    )

def process_single_image(img, settings=None):
    """Main processing pipeline"""
    if settings is None:
        settings = current_settings
    
    if settings.get('passthrough_mode', False):
        return img.copy()
    
    processed = img.copy()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING IMAGE")
    print(f"  Input: {processed.shape}, {processed.dtype}")
    
    # Step 1: Auto-straighten
    if settings['auto_straighten']:
        angle = detect_rotation(processed)
        if abs(angle) > settings['straighten_threshold']:
            processed = rotate_image(processed, angle)
    
    # Step 2: Remove borders
    if settings['remove_borders']:
        processed = remove_borders(
            processed,
            settings['border_threshold_low'],
            settings['border_threshold_high'],
            settings.get('border_buffer_percent', 1)
        )
    
    # Step 3: White balance
    if settings['auto_white_balance']:
        processed = auto_white_balance(processed, settings['white_balance_strength'])
    
    print(f"  Output: {processed.shape}, {processed.dtype}")
    print(f"{'='*60}\n")
    
    return processed


def extract_color_features(img):
    """Extract color features for clustering"""
    # Convert to 8-bit for feature extraction
    if img.dtype == np.uint16:
        small = cv2.resize(img, (100, 100))
        small = (small / 256).astype(np.uint8)
    else:
        small = cv2.resize(img, (100, 100))
    
    # Convert to LAB
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    
    features = []
    for i in range(3):
        features.append(np.mean(lab[:, :, i]))
        features.append(np.std(lab[:, :, i]))
    
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
    
    return np.array(features)


def img_to_base64(img):
    """Convert image to base64 for web display"""
    # Convert to 8-bit for display
    if img.dtype == np.uint16:
        img_8bit = (img / 256).astype(np.uint8)
    else:
        img_8bit = img
    
    _, buffer = cv2.imencode('.jpg', img_8bit, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def cluster_images_by_color(images_data, max_rolls=38):
    """Cluster images into rolls by color similarity"""
    if not images_data:
        return []
    
    features = np.array([img['features'] for img in images_data])
    
    # Determine number of clusters
    n_clusters = min(len(images_data), max_rolls)
    if n_clusters == 1:
        return [{
            'roll_number': 1,
            'photo_count': len(images_data),
            'photos': [{'image_id': img['image_id'], 'filename': img['filename'], 
                       'image_data': img_to_base64(img['img'])} for img in images_data]
        }]
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Group by cluster
    rolls_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        rolls_dict[label].append(images_data[idx])
    
    # Create roll objects
    rolls = []
    for roll_num, (label, imgs) in enumerate(sorted(rolls_dict.items()), start=1):
        rolls.append({
            'roll_number': roll_num,
            'photo_count': len(imgs),
            'photos': [{'image_id': img['image_id'], 'filename': img['filename'],
                       'image_data': img_to_base64(img['img'])} for img in imgs]
        })
    
    return rolls


# -----------------------------
# Flask Routes
# -----------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get-output-folder', methods=['GET'])
def get_output_folder():
    return jsonify({'path': app.config['OUTPUT_BASE']})


@app.route('/set-output-folder', methods=['POST'])
def set_output_folder():
    data = request.json
    new_path = data.get('path')
    if not new_path:
        return jsonify({'error': 'No path provided'}), 400
    try:
        resolved_path = Path(new_path).resolve()
        resolved_path.mkdir(parents=True, exist_ok=True)
        app.config['OUTPUT_BASE'] = str(resolved_path)
        return jsonify({'success': True, 'path': str(resolved_path)})
    except Exception:
        return jsonify({'error': 'Invalid path'}), 400


@app.route('/update-settings', methods=['POST'])
def update_settings():
    """Update processing settings from frontend"""
    global current_settings
    data = request.json
    if data:
        current_settings.update(data)
        print(f"Settings updated: {current_settings}")
        return jsonify({'success': True})
    return jsonify({'error': 'No settings provided'}), 400


@app.route('/progress', methods=['GET'])
def get_progress():
    def generate():
        while True:
            if progress_data['total'] > 0:
                percent = int((progress_data['current'] / progress_data['total']) * 100)
                elapsed = time.time() - progress_data['start_time']
                if progress_data['current'] > 0:
                    rate = progress_data['current'] / elapsed
                    remaining = (progress_data['total'] - progress_data['current']) / rate
                    time_remaining = f"{int(remaining // 60)}m {int(remaining % 60)}s"
                else:
                    time_remaining = "Calculating..."
                yield f"data: {json.dumps({**progress_data, 'percent': percent, 'time_remaining': time_remaining})}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')


@app.route('/reprocess-image', methods=['POST'])
def reprocess_image():
    data = request.json
    image_id = data.get('image_id')
    settings = data.get('settings', current_settings)
    if image_id not in image_cache:
        return jsonify({'error': 'Image not found'}), 404

    original_img = image_cache[image_id]['original'].copy()
    rotation = data.get('rotation', 0)
    if rotation != 0:
        original_img = rotate_image(original_img, rotation)

    processed = process_single_image(original_img, settings)
    image_cache[image_id]['processed'] = processed

    return jsonify({'success': True, 'image_data': img_to_base64(processed)})


@app.route('/process', methods=['POST'])
def process_photos():
    global image_cache, progress_data
    image_cache = {}

    try:
        files = request.files.getlist('photos')
        save_files = request.form.get('saveFiles', 'true') == 'true'
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        progress_data.update({'current': 0, 'total': len(files), 'start_time': time.time(), 'status': 'Starting...'})

        images_data = []
        raw_formats = ('.arw', '.cr2', '.nef', '.dng', '.raf', '.orf', '.rw2', '.raw')
        standard_formats = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        all_formats = raw_formats + standard_formats

        for idx, file in enumerate(files):
            progress_data['current'] = idx + 1
            progress_data['current_file'] = file.filename
            progress_data['status'] = f'Processing {file.filename}...'

            if not file.filename.lower().endswith(all_formats):
                print(f"Skipping unsupported file: {file.filename}")
                continue

            try:
                img_bytes = file.read()
                metadata = {}
                
                if file.filename.lower().endswith(raw_formats):
                    print(f"Loading RAW: {file.filename}")
                    img, metadata = load_raw_image(img_bytes)
                elif file.filename.lower().endswith(('.tif', '.tiff')):
                    print(f"Loading TIFF: {file.filename}")
                    img, metadata = load_tiff_safe(img_bytes)
                else:
                    print(f"Loading standard image: {file.filename}")
                    # Try to extract EXIF from standard images
                    try:
                        import io
                        from PIL import Image as PILImage
                        from PIL.ExifTags import TAGS
                        pil_img = PILImage.open(io.BytesIO(img_bytes))
                        exif_data = pil_img.getexif()
                        if exif_data:
                            for tag_id, value in exif_data.items():
                                tag = TAGS.get(tag_id, tag_id)
                                metadata[tag] = value
                    except:
                        pass
                    
                    img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    img = img_bgr

                if img is None:
                    print(f"Failed to load: {file.filename}")
                    continue

                safe_name = Path(file.filename).name
                image_id = f"img_{idx}_{safe_name}"
                
                processed = process_single_image(img, current_settings)
                
                image_cache[image_id] = {
                    'original': img, 
                    'processed': processed, 
                    'filename': safe_name,
                    'metadata': metadata
                }

                features = extract_color_features(processed)
                images_data.append({'image_id': image_id, 'filename': safe_name, 'img': processed, 'features': features})
            
            except Exception as e:
                print(f"Error processing {file.filename}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        if not images_data:
            progress_data['status'] = 'Error: No images processed'
            return jsonify({'error': 'No images were successfully processed'}), 400

        print(f"Successfully processed {len(images_data)} images")

        # Cluster images
        progress_data['status'] = 'Grouping images...'
        rolls = cluster_images_by_color(images_data, max_rolls=38)
        print(f"Grouped into {len(rolls)} rolls")

        # Save files
        output_path = None
        if save_files:
            Path(app.config['OUTPUT_BASE']).mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_folder = Path(app.config['OUTPUT_BASE']) / f'session_{timestamp}'
            session_folder.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving to: {session_folder}")

            for roll in rolls:
                roll_folder = session_folder / f"roll_{roll['roll_number']:02d}"
                roll_folder.mkdir(parents=True, exist_ok=True)
                
                print(f"Saving Roll {roll['roll_number']} with {roll['photo_count']} photos")
                
                for photo in roll['photos']:
                    img_data = image_cache[photo['image_id']]['processed']
                    metadata = image_cache[photo['image_id']].get('metadata', {})
                    output_file = roll_folder / (Path(photo['filename']).stem + '.tiff')
                    
                    # Save 16-bit TIFF
                    try:
                        is_16bit = img_data.dtype == np.uint16
                        
                        print(f"    Saving {photo['filename']}: shape={img_data.shape}, dtype={img_data.dtype}")
                        if metadata:
                            print(f"    Embedding {len(metadata)} EXIF tags")
                        
                        # Convert BGR to RGB for saving
                        if img_data.shape[2] == 3:
                            rgb_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                        else:
                            rgb_data = img_data
                        
                        # Apply gamma curve for proper display if 16-bit linear
                        if is_16bit:
                            rgb_float = rgb_data.astype(np.float64) / 65535.0
                            rgb_gamma = np.power(rgb_float, 1.0/2.2)
                            rgb_data = (rgb_gamma * 65535.0).astype(np.uint16)
                        
                        # Use tifffile for proper 16-bit RGB TIFF support
                        import tifffile
                        
                        # Save with tifffile first
                        save_kwargs = {
                            'photometric': 'rgb'
                        }
                        
                        # Create description with key metadata
                        description_parts = []
                        if metadata:
                            if 'Make' in metadata:
                                description_parts.append(f"Make: {metadata['Make']}")
                            if 'Model' in metadata:
                                description_parts.append(f"Model: {metadata['Model']}")
                            if 'DateTime' in metadata or 'DateTimeOriginal' in metadata:
                                dt = metadata.get('DateTimeOriginal', metadata.get('DateTime', ''))
                                description_parts.append(f"Date: {dt}")
                            if 'ExposureTime' in metadata:
                                description_parts.append(f"Exposure: {metadata['ExposureTime']}")
                            if 'FNumber' in metadata:
                                description_parts.append(f"F-stop: f/{metadata['FNumber']}")
                            if 'ISOSpeedRatings' in metadata or 'ISO' in metadata:
                                iso = metadata.get('ISOSpeedRatings', metadata.get('ISO', ''))
                                description_parts.append(f"ISO: {iso}")
                            if 'FocalLength' in metadata:
                                description_parts.append(f"Focal Length: {metadata['FocalLength']}mm")
                            if 'LensModel' in metadata:
                                description_parts.append(f"Lens: {metadata['LensModel']}")
                            
                            if description_parts:
                                save_kwargs['description'] = ' | '.join(description_parts)
                                print(f"    Metadata: {save_kwargs['description']}")
                        
                        # Use compression if enabled
                        use_compression = current_settings.get('use_tiff_compression', False)
                        if use_compression:
                            save_kwargs['compression'] = 'lzw'
                        
                        try:
                            tifffile.imwrite(str(output_file), rgb_data, **save_kwargs)
                            comp_str = " (compressed)" if use_compression else " (uncompressed)"
                            
                            # Now embed proper EXIF using piexif (for Windows Camera section)
                            if metadata:
                                try:
                                    import piexif
                                    from PIL import Image as PILImage
                                    
                                    # Build EXIF dictionary
                                    exif_dict = {"0th": {}, "Exif": {}}
                                    
                                    # 0th IFD (Image)
                                    if 'Make' in metadata:
                                        make_val = str(metadata['Make'])
                                        exif_dict["0th"][piexif.ImageIFD.Make] = make_val.encode('utf-8')
                                        print(f"      Make: {make_val}")
                                    
                                    if 'Model' in metadata:
                                        model_val = str(metadata['Model'])
                                        exif_dict["0th"][piexif.ImageIFD.Model] = model_val.encode('utf-8')
                                        print(f"      Model: {model_val}")
                                    
                                    if 'DateTime' in metadata:
                                        dt_val = str(metadata['DateTime'])
                                        exif_dict["0th"][piexif.ImageIFD.DateTime] = dt_val.encode('utf-8')
                                    
                                    # Exif IFD (Camera settings)
                                    if 'DateTimeOriginal' in metadata:
                                        dto_val = str(metadata['DateTimeOriginal'])
                                        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = dto_val.encode('utf-8')
                                    
                                    # Exposure time (needs to be fraction)
                                    if 'ExposureTime' in metadata:
                                        try:
                                            exp = str(metadata['ExposureTime'])
                                            if '/' in exp:
                                                parts = exp.split('/')
                                                num, denom = int(parts[0].strip()), int(parts[1].strip())
                                                exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (num, denom)
                                                print(f"      Exposure: {num}/{denom}")
                                            else:
                                                exp_float = float(exp)
                                                exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (int(exp_float * 1000), 1000)
                                                print(f"      Exposure: {exp_float}s")
                                        except Exception as e:
                                            print(f"      Warning: Could not parse ExposureTime: {e}")
                                    
                                    # F-Number (needs to be fraction)
                                    if 'FNumber' in metadata:
                                        try:
                                            f_str = str(metadata['FNumber'])
                                            # Remove 'f/' prefix if present
                                            f_str = f_str.replace('f/', '').replace('F/', '').strip()
                                            
                                            if '/' in f_str:
                                                parts = f_str.split('/')
                                                num, denom = int(parts[0].strip()), int(parts[1].strip())
                                                exif_dict["Exif"][piexif.ExifIFD.FNumber] = (num, denom)
                                            else:
                                                f_val = float(f_str)
                                                exif_dict["Exif"][piexif.ExifIFD.FNumber] = (int(f_val * 10), 10)
                                            print(f"      F-stop: f/{f_str}")
                                        except Exception as e:
                                            print(f"      Warning: Could not parse FNumber: {e}")
                                    
                                    # ISO
                                    if 'ISOSpeedRatings' in metadata or 'ISO' in metadata:
                                        try:
                                            iso_val = metadata.get('ISOSpeedRatings', metadata.get('ISO', 0))
                                            iso_int = int(str(iso_val).strip())
                                            exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings] = iso_int
                                            print(f"      ISO: {iso_int}")
                                        except Exception as e:
                                            print(f"      Warning: Could not parse ISO: {e}")
                                    
                                    # Focal Length (needs to be fraction)
                                    if 'FocalLength' in metadata:
                                        try:
                                            fl_str = str(metadata['FocalLength'])
                                            # Remove 'mm' suffix if present
                                            fl_str = fl_str.replace('mm', '').strip()
                                            
                                            if '/' in fl_str:
                                                parts = fl_str.split('/')
                                                num, denom = int(parts[0].strip()), int(parts[1].strip())
                                                exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (num, denom)
                                            else:
                                                fl_val = float(fl_str)
                                                exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (int(fl_val * 10), 10)
                                            print(f"      Focal Length: {fl_str}mm")
                                        except Exception as e:
                                            print(f"      Warning: Could not parse FocalLength: {e}")
                                    
                                    # Lens Model
                                    if 'LensModel' in metadata:
                                        try:
                                            lens_val = str(metadata['LensModel'])
                                            exif_dict["Exif"][piexif.ExifIFD.LensModel] = lens_val.encode('utf-8')
                                            print(f"      Lens: {lens_val}")
                                        except Exception as e:
                                            print(f"      Warning: Could not parse LensModel: {e}")
                                    
                                    # Create EXIF bytes
                                    print(f"      Creating EXIF bytes...")
                                    exif_bytes = piexif.dump(exif_dict)
                                    
                                    # Re-save TIFF with EXIF using PIL
                                    # piexif.insert() only works with JPEG, not TIFF
                                    # So we reload and re-save with PIL
                                    print(f"      Re-saving TIFF with EXIF...")
                                    pil_img = PILImage.open(str(output_file))
                                    pil_img.save(
                                        str(output_file),
                                        exif=exif_bytes,
                                        compression='tiff_lzw' if use_compression else None
                                    )
                                    
                                    print(f"    ✓ Saved {output_file.name}{comp_str} with EXIF")
                                    
                                except ImportError as e:
                                    print(f"    ✓ Saved {output_file.name}{comp_str} (piexif not installed: {e})")
                                except Exception as e:
                                    import traceback
                                    print(f"    ✓ Saved {output_file.name}{comp_str} (EXIF embed failed)")
                                    print(f"      Error: {type(e).__name__}: {str(e)}")
                                    traceback.print_exc()
                            else:
                                print(f"    ✓ Saved {output_file.name}{comp_str}")
                                
                        except Exception as e:
                            if 'compression' in save_kwargs:
                                # Retry without compression
                                print(f"    Warning: Compression failed ({e}), saving uncompressed")
                                del save_kwargs['compression']
                                tifffile.imwrite(str(output_file), rgb_data, **save_kwargs)
                                print(f"    ✓ Saved {output_file.name} (uncompressed)")
                            else:
                                raise
                    except Exception as e:
                        print(f"  Error saving {output_file.name}: {e}")
            
            output_path = str(session_folder)

        progress_data['status'] = 'Complete!'
        
        response = {'rolls': rolls}
        if output_path:
            response['output_path'] = output_path
        
        return jsonify(response)

    except Exception as e:
        progress_data['status'] = f'Error: {str(e)}'
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3683, debug=False, threaded=True)