"""
Flask Photo Processor Web App - Enhanced Version
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import rawpy
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
import base64
import io
import zipfile
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Create processed directory
PROCESSED_DIR = Path('/app/processed')
PROCESSED_DIR.mkdir(exist_ok=True)


def read_image(file_bytes, filename):
    """Read image from bytes, supporting RAW formats"""
    ext = Path(filename).suffix.lower()
    
    # Handle RAW formats
    if ext in ['.arw', '.cr2', '.nef', '.dng', '.raf', '.orf', '.rw2']:
        try:
            with rawpy.imread(io.BytesIO(file_bytes)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8
                )
            # Convert RGB to BGR for OpenCV
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"RAW processing failed for {filename}: {e}")
            return None
    
    # Handle standard formats
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is not None:
        # Check and apply EXIF orientation
        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            exif = pil_img.getexif()
            if exif:
                orientation = exif.get(274)  # 274 is orientation tag
                if orientation == 3:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                elif orientation == 6:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif orientation == 8:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except:
            pass
    
    return img


def auto_white_balance(img, is_16bit=False):
    """Apply auto white balance"""
    # Scale factor for conversion
    max_val = 65535 if is_16bit else 255
    
    # Convert to LAB
    if is_16bit:
        # Normalize to 8-bit for LAB conversion
        img_8bit = (img / 256).astype(np.uint8)
        result = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2LAB)
    else:
        result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    
    if is_16bit:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        # Scale back to 16-bit
        return (result_rgb.astype(np.float32) * 256).astype(np.uint16)
    else:
        return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


def extract_color_signature(img, is_16bit=False):
    """Extract color signature for clustering"""
    # Work with 8-bit for speed
    if is_16bit:
        small = cv2.resize((img / 256).astype(np.uint8), (100, 100))
    else:
        small = cv2.resize(img, (100, 100))
    
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    features = []
    for i in range(3):
        features.append(np.mean(lab[:, :, i]))
        features.append(np.std(lab[:, :, i]))
    pixels = small.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans.fit(pixels)
    features.extend(kmeans.cluster_centers_[0])
    return np.array(features)


def remove_borders(img, is_16bit=False, threshold_low=15, threshold_high=240, buffer_percent=0.05):
    """Remove black and white borders with 5% buffer inward"""
    # Convert to grayscale for detection
    if is_16bit:
        gray = cv2.cvtColor((img / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    h, w = gray.shape
    
    # Create mask for both dark and bright borders
    _, thresh_dark = cv2.threshold(gray, threshold_low, 255, cv2.THRESH_BINARY)
    _, thresh_bright = cv2.threshold(gray, threshold_high, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_and(thresh_dark, thresh_bright)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # Get the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w_crop, h_crop = cv2.boundingRect(largest)
    
    # Only crop if the detected region is significant
    if w_crop > w * 0.5 and h_crop > h * 0.5:
        # Calculate 5% buffer to crop inward from detected edges
        buffer_x = int(w_crop * buffer_percent)
        buffer_y = int(h_crop * buffer_percent)
        
        # Apply buffer inward from detected boundaries
        x = max(0, x + buffer_x)
        y = max(0, y + buffer_y)
        w_crop = max(1, w_crop - 2 * buffer_x)
        h_crop = max(1, h_crop - 2 * buffer_y)
        
        # Ensure we don't go outside image bounds
        w_crop = min(w - x, w_crop)
        h_crop = min(h - y, h_crop)
        
        return img[y:y+h_crop, x:x+w_crop]
    
    return img


def detect_rotation(img, is_16bit=False):
    """Detect rotation angle with improved algorithm"""
    # Use 8-bit for edge detection
    if is_16bit:
        gray = cv2.cvtColor((img / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    h, w = gray.shape
    
    # Use edges for detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=min(w, h) * 0.3, maxLineGap=20)
    
    if lines is None:
        return 0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Normalize angle to -45 to 45 range
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        # Only consider significant angles
        if abs(angle) > 0.5:
            angles.append(angle)
    
    if angles:
        # Use median to avoid outliers
        median_angle = np.median(angles)
        # Only return angle if it's significant
        if abs(median_angle) > 0.5 and abs(median_angle) < 10:
            return -median_angle
    
    return 0


def rotate_image(img, angle, is_16bit=False):
    """Rotate image with proper border handling"""
    if abs(angle) < 0.1:
        return img
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust translation
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    
    # Rotate with white background
    border_val = 65535 if is_16bit else 255
    rotated = cv2.warpAffine(img, matrix, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(border_val, border_val, border_val))
    
    return rotated


def img_to_base64(img, format='jpg'):
    """Convert image to base64 for preview (compressed)"""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def save_image(img, filename, output_dir):
    """Save processed image in original format with maximum quality"""
    original_ext = Path(filename).suffix.lower()
    output_path = output_dir / filename
    
    # Handle different formats with maximum quality
    if original_ext in ['.jpg', '.jpeg']:
        # Maximum quality JPEG
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif original_ext == '.png':
        # Lossless PNG with maximum compression
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    elif original_ext == '.tiff' or original_ext == '.tif':
        # Lossless TIFF
        cv2.imwrite(str(output_path), img)
    else:
        # Default to PNG for RAW conversions (lossless)
        output_path = output_dir / (Path(filename).stem + '.png')
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_photos():
    try:
        files = request.files.getlist('photos')
        save_mode = request.form.get('save_mode', 'memory')  # memory, new_location, overwrite
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        images_data = []
        features_list = []
        total_files = len(files)
        
        # Process each image
        for idx, file in enumerate(files):
            # Yield progress
            progress = {
                'current': idx + 1,
                'total': total_files,
                'filename': file.filename,
                'action': 'Reading file...'
            }
            
            img_bytes = file.read()
            img = read_image(img_bytes, file.filename)
            
            if img is not None:
                original_img = img.copy()
                
                # Extract features before processing
                features = extract_color_signature(img)
                features_list.append(features)
                
                # Process image
                progress['action'] = 'Detecting rotation...'
                angle = detect_rotation(img)
                
                if abs(angle) > 0.5:
                    progress['action'] = f'Rotating {angle:.1f}°...'
                    img = rotate_image(img, angle)
                
                progress['action'] = 'Removing borders...'
                img = remove_borders(img)
                
                progress['action'] = 'Applying white balance...'
                img = auto_white_balance(img)
                
                images_data.append({
                    'filename': file.filename,
                    'img': img,
                    'features': features
                })
        
        if len(images_data) < 2:
            # Create full quality versions for download
            images_map = {}
            for d in images_data:
                original_ext = Path(d['filename']).suffix.lower()
                if original_ext in ['.jpg', '.jpeg']:
                    _, buffer = cv2.imencode('.jpg', d['img'], [cv2.IMWRITE_JPEG_QUALITY, 100])
                elif original_ext == '.png':
                    _, buffer = cv2.imencode('.png', d['img'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                elif original_ext in ['.arw', '.cr2', '.nef', '.dng', '.raf', '.orf', '.rw2']:
                    # RAW files saved as PNG
                    _, buffer = cv2.imencode('.png', d['img'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                else:
                    _, buffer = cv2.imencode('.png', d['img'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                images_map[d['filename']] = base64.b64encode(buffer).decode('utf-8')
            
            rolls = [{
                'roll_number': 1,
                'photos': [
                    {'filename': d['filename'], 'data': img_to_base64(d['img'])}
                    for d in images_data
                ]
            }]
            return jsonify({'rolls': rolls, 'images_map': images_map})
        
        # Cluster images
        features_array = np.array(features_list)
        total_photos = len(features_array)
        estimated_rolls = max(1, round(total_photos / 30))
        n_clusters = max(1, min(estimated_rolls, total_photos))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_array)
        
        groups = defaultdict(list)
        for idx, label in enumerate(labels):
            groups[int(label)].append(images_data[idx])
        
        # Create full quality versions for download
        images_map = {}
        for idx, img_data in enumerate(images_data):
            original_ext = Path(img_data['filename']).suffix.lower()
            if original_ext in ['.jpg', '.jpeg']:
                _, buffer = cv2.imencode('.jpg', img_data['img'], [cv2.IMWRITE_JPEG_QUALITY, 100])
            elif original_ext == '.png':
                _, buffer = cv2.imencode('.png', img_data['img'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            elif original_ext in ['.arw', '.cr2', '.nef', '.dng', '.raf', '.orf', '.rw2']:
                # RAW files saved as PNG (lossless)
                _, buffer = cv2.imencode('.png', img_data['img'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                _, buffer = cv2.imencode('.png', img_data['img'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            images_map[img_data['filename']] = base64.b64encode(buffer).decode('utf-8')
        
        rolls = []
        for roll_id, photos in sorted(groups.items()):
            rolls.append({
                'roll_number': roll_id + 1,
                'photo_count': len(photos),
                'photos': [
                    {'filename': p['filename'], 'data': img_to_base64(p['img'])}
                    for p in photos
                ]
            })
        
        return jsonify({'rolls': rolls, 'images_map': images_map})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download', methods=['POST'])
def download_processed():
    """Create a zip file of all processed images in full quality"""
    try:
        data = request.json
        rolls = data.get('rolls', [])
        images_map = data.get('images_map', {})
        
        # Create zip in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED) as zip_file:
            for roll in rolls:
                roll_num = roll['roll_number']
                for photo in roll['photos']:
                    filename = photo['filename']
                    original_ext = Path(filename).suffix.lower()
                    
                    # Get the full quality image data
                    img_bytes = images_map.get(filename)
                    if img_bytes:
                        # Decode base64 full quality image
                        img_data = base64.b64decode(img_bytes)
                    else:
                        # Fallback to preview (shouldn't happen)
                        img_data = base64.b64decode(photo['data'])
                    
                    # For RAW files converted to PNG, update extension
                    if original_ext in ['.arw', '.cr2', '.nef', '.dng', '.raf', '.orf', '.rw2']:
                        filename = Path(filename).stem + '.png'
                    
                    # Add to zip with no compression to preserve quality
                    zip_file.writestr(
                        f'roll_{roll_num}/{filename}',
                        img_data
                    )
        
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'processed_photos_{timestamp}.zip'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Film Roll Photo Processor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #1a1a1a;
            padding: 20px;
            color: #e0e0e0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: #2d2d2d;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        h1 {
            color: #fff;
            margin-bottom: 30px;
            font-size: 28px;
        }
        .upload-section {
            margin-bottom: 30px;
            padding: 20px;
            border: 2px dashed #555;
            border-radius: 8px;
            text-align: center;
            background: #242424;
        }
        .upload-section h2 {
            color: #e0e0e0;
            margin-bottom: 15px;
        }
        input[type="file"] {
            margin: 10px 0;
            color: #e0e0e0;
        }
        .button-group {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 15px;
        }
        button {
            background: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
        }
        button:hover {
            background: #45a049;
            transform: translateY(-1px);
        }
        button:disabled {
            background: #555;
            cursor: not-allowed;
            transform: none;
        }
        .download-btn {
            background: #2196F3;
        }
        .download-btn:hover {
            background: #1976D2;
        }
        .spinner {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .spinner.active {
            display: block;
        }
        .spinner-icon {
            border: 4px solid #333;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner-text {
            color: #aaa;
            font-size: 16px;
        }
        .results {
            margin-top: 30px;
        }
        .roll {
            margin-bottom: 40px;
            border: 1px solid #444;
            border-radius: 8px;
            overflow: hidden;
            background: #242424;
        }
        .roll-header {
            background: #333;
            padding: 15px 20px;
            border-bottom: 1px solid #444;
        }
        .roll-title {
            font-size: 20px;
            font-weight: 600;
            color: #fff;
        }
        .roll-count {
            font-size: 14px;
            color: #aaa;
            margin-top: 5px;
        }
        .photos-container {
            display: flex;
            overflow-x: auto;
            padding: 20px;
            gap: 15px;
            background: #242424;
        }
        .photos-container::-webkit-scrollbar {
            height: 10px;
        }
        .photos-container::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        .photos-container::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 5px;
        }
        .photo {
            flex-shrink: 0;
            width: 250px;
            border: 1px solid #444;
            border-radius: 5px;
            overflow: hidden;
            background: #333;
        }
        .photo img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            display: block;
        }
        .photo-name {
            padding: 10px;
            font-size: 12px;
            color: #aaa;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .error {
            color: #ff5252;
            padding: 15px;
            background: #3d1f1f;
            border-radius: 5px;
            margin: 20px 0;
            border: 1px solid #ff5252;
        }
        
        /* Progress Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal-overlay.active {
            display: flex;
        }
        .modal {
            background: #2d2d2d;
            border-radius: 10px;
            padding: 30px;
            max-width: 500px;
            width: 90%;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border: 1px solid #444;
        }
        .modal-header {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #fff;
            text-align: center;
        }
        .progress-info {
            margin-bottom: 20px;
        }
        .current-file {
            font-size: 14px;
            color: #aaa;
            margin-bottom: 5px;
        }
        .current-action {
            font-size: 16px;
            color: #4CAF50;
            font-weight: 500;
            margin-bottom: 15px;
        }
        .progress-stats {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 14px;
            color: #aaa;
        }
        .progress-bar-container {
            width: 100%;
            height: 30px;
            background: #1a1a1a;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 15px;
            border: 1px solid #444;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            width: 0%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 14px;
        }
        .time-remaining {
            text-align: center;
            color: #64b5f6;
            font-size: 14px;
        }
        .processing-steps {
            margin-top: 20px;
            padding: 15px;
            background: #242424;
            border-radius: 5px;
            border: 1px solid #444;
        }
        .step {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            font-size: 13px;
            color: #aaa;
        }
        .step:last-child {
            margin-bottom: 0;
        }
        .step-icon {
            width: 20px;
            margin-right: 10px;
        }
        .step.active {
            color: #4CAF50;
        }
        .step.complete {
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📷 Film Roll Photo Processor</h1>
        
        <div class="upload-section">
            <h2>Select Photos</h2>
            <p style="color: #aaa; font-size: 14px; margin-bottom: 10px;">Supports JPG, PNG, ARW, CR2, NEF, DNG, and other RAW formats</p>
            <input type="file" id="photoInput" multiple accept="image/*,.arw,.cr2,.nef,.dng,.raf,.orf,.rw2" />
            <div class="button-group">
                <button id="processBtn" onclick="processPhotos()">Process Photos</button>
                <button id="downloadBtn" class="download-btn" onclick="downloadPhotos()" style="display:none;">Download All</button>
            </div>
        </div>
        
        <div class="spinner" id="spinner">
            <div class="spinner-icon"></div>
            <div class="spinner-text">Processing photos... This may take a few minutes.</div>
        </div>
        
        <div id="error"></div>
        <div id="results" class="results"></div>
    </div>

    <script>
        let processedRolls = null;
        let imagesMap = null;
        
        async function processPhotos() {
            const input = document.getElementById('photoInput');
            const files = input.files;
            
            if (!files || files.length === 0) {
                alert('Please select photos first');
                return;
            }
            
            const formData = new FormData();
            for (let file of files) {
                formData.append('photos', file);
            }
            formData.append('save_mode', 'memory');
            
            document.getElementById('spinner').classList.add('active');
            document.getElementById('processBtn').disabled = true;
            document.getElementById('error').innerHTML = '';
            document.getElementById('results').innerHTML = '';
            document.getElementById('downloadBtn').style.display = 'none';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('error').innerHTML = 
                        `<div class="error">Error: ${data.error}</div>`;
                    return;
                }
                
                processedRolls = data.rolls;
                imagesMap = data.images_map;
                displayRolls(data.rolls);
                document.getElementById('downloadBtn').style.display = 'block';
                
            } catch (error) {
                document.getElementById('error').innerHTML = 
                    `<div class="error">Error: ${error.message}</div>`;
            } finally {
                document.getElementById('spinner').classList.remove('active');
                document.getElementById('processBtn').disabled = false;
            }
        }
        
        async function downloadPhotos() {
            if (!processedRolls || !imagesMap) return;
            
            try {
                const response = await fetch('/download', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        rolls: processedRolls,
                        images_map: imagesMap
                    })
                });
                
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `processed_photos_${new Date().getTime()}.zip`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
            } catch (error) {
                document.getElementById('error').innerHTML = 
                    `<div class="error">Download error: ${error.message}</div>`;
            }
        }
        
        function displayRolls(rolls) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';
            
            rolls.forEach(roll => {
                const rollDiv = document.createElement('div');
                rollDiv.className = 'roll';
                
                rollDiv.innerHTML = `
                    <div class="roll-header">
                        <div class="roll-title">🎞️ Film Roll ${roll.roll_number}</div>
                        <div class="roll-count">${roll.photo_count} photos</div>
                    </div>
                    <div class="photos-container">
                        ${roll.photos.map(photo => `
                            <div class="photo">
                                <img src="data:image/jpeg;base64,${photo.data}" alt="${photo.filename}">
                                <div class="photo-name">${photo.filename}</div>
                            </div>
                        `).join('')}
                    </div>
                `;
                
                resultsDiv.appendChild(rollDiv);
            });
        }
    </script>
</body>
</html>
'''

@app.before_request
def create_template():
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    template_file = templates_dir / 'index.html'
    template_file.write_text(HTML_TEMPLATE)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3683, debug=False)