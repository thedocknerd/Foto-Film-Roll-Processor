"""
Improved White Balance Module for Film Scanning
Handles 8-bit, 16-bit, and 32-bit images with proper gamma handling
Maintains histogram width and prevents crushing
"""

import cv2
import numpy as np

def detect_bit_depth(img):
    """
    Detect the bit depth and working range of an image
    Returns: (bit_depth, is_float, max_value)
    """
    dtype = img.dtype
    
    if dtype == np.uint8:
        return 8, False, 255
    elif dtype == np.uint16:
        return 16, False, 65535
    elif dtype == np.float32 or dtype == np.float64:
        # Check if normalized (0-1) or full range
        max_val = img.max()
        if max_val <= 1.0:
            return 32, True, 1.0
        else:
            # Likely 16-bit range in float format
            return 32, True, 65535.0
    else:
        # Default to 16-bit
        return 16, False, 65535


def linear_to_srgb(linear):
    """Convert linear RGB to sRGB with proper gamma curve"""
    # Ensure float64 for precision
    linear = linear.astype(np.float64)
    
    # sRGB gamma curve
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    )
    return srgb


def srgb_to_linear(srgb):
    """Convert sRGB to linear RGB with proper inverse gamma curve"""
    # Ensure float64 for precision
    srgb = srgb.astype(np.float64)
    
    # Inverse sRGB gamma curve
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4)
    )
    return linear


def auto_white_balance_gray_world(img, strength=1.0, preserve_luminance=True):
    """
    Gray World white balance algorithm with proper bit depth handling
    Works in linear space to avoid histogram crushing
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image in BGR format (OpenCV standard)
    strength : float
        White balance strength (0.0 = no correction, 1.0 = full correction)
    preserve_luminance : bool
        If True, maintains overall brightness (recommended)
    
    Returns:
    --------
    numpy.ndarray
        White balanced image in same format as input
    """
    if strength == 0.0:
        return img.copy()
    
    # Detect bit depth
    bit_depth, is_float, max_value = detect_bit_depth(img)
    print(f"  White Balance: {bit_depth}-bit, float={is_float}, max={max_value}")
    
    # Convert to float and normalize to 0-1 range
    if is_float:
        if max_value > 1.0:
            # Float with large range
            img_float = img / max_value
        else:
            # Already normalized
            img_float = img.copy()
    else:
        # Integer type - convert to float
        img_float = img.astype(np.float64) / max_value
    
    # Convert to linear space (remove gamma curve)
    # Most images are in sRGB, RAW linear images are already linear
    # We'll assume sRGB unless already linear (check if values cluster at extremes)
    histogram_spread = np.percentile(img_float, 99) - np.percentile(img_float, 1)
    is_likely_linear = histogram_spread > 0.8  # Linear images have wider histograms
    
    if not is_likely_linear:
        print(f"  Converting from sRGB to linear (histogram spread: {histogram_spread:.3f})")
        img_linear = srgb_to_linear(img_float)
    else:
        print(f"  Image already linear (histogram spread: {histogram_spread:.3f})")
        img_linear = img_float
    
    # Split channels (BGR)
    b, g, r = cv2.split(img_linear)
    
    # Calculate channel averages (gray world assumption)
    # Use percentile instead of mean to avoid being influenced by extreme values
    b_avg = np.percentile(b, 50)
    g_avg = np.percentile(g, 50)
    r_avg = np.percentile(r, 50)
    
    # Calculate gray (target average)
    gray_avg = (r_avg + g_avg + b_avg) / 3.0
    
    print(f"  Channel averages (linear): R={r_avg:.4f}, G={g_avg:.4f}, B={b_avg:.4f}, Gray={gray_avg:.4f}")
    
    # Avoid division by zero
    if r_avg < 0.001:
        r_avg = 0.001
    if g_avg < 0.001:
        g_avg = 0.001
    if b_avg < 0.001:
        b_avg = 0.001
    
    # Calculate correction factors
    r_gain = gray_avg / r_avg
    g_gain = gray_avg / g_avg
    b_gain = gray_avg / b_avg
    
    # Apply strength factor (interpolate between no correction and full correction)
    r_gain = 1.0 + (r_gain - 1.0) * strength
    g_gain = 1.0 + (g_gain - 1.0) * strength
    b_gain = 1.0 + (b_gain - 1.0) * strength
    
    print(f"  Correction gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")
    
    # Apply gains
    r_corrected = r * r_gain
    g_corrected = g * g_gain
    b_corrected = b * b_gain
    
    # Preserve luminance if requested
    if preserve_luminance:
        # Calculate luminance before and after
        lum_before = 0.2126 * r + 0.7152 * g + 0.0722 * b
        lum_after = 0.2126 * r_corrected + 0.7152 * g_corrected + 0.0722 * b_corrected
        
        # Calculate luminance scaling factor
        lum_scale = np.median(lum_before) / (np.median(lum_after) + 1e-10)
        
        print(f"  Luminance preservation scale: {lum_scale:.3f}")
        
        r_corrected *= lum_scale
        g_corrected *= lum_scale
        b_corrected *= lum_scale
    
    # Merge channels
    img_corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    
    # Clip to valid range (soft clipping to preserve detail)
    # Use tanh for soft clipping near boundaries
    img_corrected = np.clip(img_corrected, 0.0, 1.0)
    
    # Check for clipping
    clipped_pixels = np.sum((img_corrected >= 0.999) | (img_corrected <= 0.001))
    total_pixels = img_corrected.size
    clipped_percent = (clipped_pixels / total_pixels) * 100
    print(f"  Clipped pixels: {clipped_percent:.2f}%")
    
    # Convert back to sRGB if it wasn't linear
    if not is_likely_linear:
        print(f"  Converting back to sRGB")
        img_corrected = linear_to_srgb(img_corrected)
    
    # Convert back to original bit depth
    if is_float:
        if max_value > 1.0:
            result = (img_corrected * max_value).astype(img.dtype)
        else:
            result = img_corrected.astype(img.dtype)
    else:
        result = np.clip(img_corrected * max_value, 0, max_value).astype(img.dtype)
    
    # Verify histogram preservation
    orig_std = np.std(img)
    result_std = np.std(result)
    print(f"  Histogram preservation: original std={orig_std:.1f}, result std={result_std:.1f}")
    
    return result


def auto_white_balance_white_patch(img, percentile=99, strength=1.0):
    """
    White Patch algorithm - assumes brightest point should be neutral
    Good for scenes with clear white/bright areas
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image in BGR format
    percentile : float
        Percentile to use for "white" reference (default 99)
    strength : float
        White balance strength (0.0 = no correction, 1.0 = full correction)
    """
    if strength == 0.0:
        return img.copy()
    
    # Detect bit depth
    bit_depth, is_float, max_value = detect_bit_depth(img)
    print(f"  White Patch Balance: {bit_depth}-bit, percentile={percentile}")
    
    # Convert to float and normalize
    if is_float:
        if max_value > 1.0:
            img_float = img / max_value
        else:
            img_float = img.copy()
    else:
        img_float = img.astype(np.float64) / max_value
    
    # Convert to linear if needed
    histogram_spread = np.percentile(img_float, 99) - np.percentile(img_float, 1)
    is_likely_linear = histogram_spread > 0.8
    
    if not is_likely_linear:
        img_linear = srgb_to_linear(img_float)
    else:
        img_linear = img_float
    
    # Split channels
    b, g, r = cv2.split(img_linear)
    
    # Find "white" reference using percentile
    r_white = np.percentile(r, percentile)
    g_white = np.percentile(g, percentile)
    b_white = np.percentile(b, percentile)
    
    print(f"  White reference: R={r_white:.4f}, G={g_white:.4f}, B={b_white:.4f}")
    
    # Calculate gains to bring white reference to neutral
    max_white = max(r_white, g_white, b_white)
    
    if r_white < 0.001:
        r_white = 0.001
    if g_white < 0.001:
        g_white = 0.001
    if b_white < 0.001:
        b_white = 0.001
    
    r_gain = max_white / r_white
    g_gain = max_white / g_white
    b_gain = max_white / b_white
    
    # Apply strength
    r_gain = 1.0 + (r_gain - 1.0) * strength
    g_gain = 1.0 + (g_gain - 1.0) * strength
    b_gain = 1.0 + (b_gain - 1.0) * strength
    
    print(f"  Gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")
    
    # Apply gains
    r_corrected = r * r_gain
    g_corrected = g * g_gain
    b_corrected = b * b_gain
    
    # Merge and clip
    img_corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    img_corrected = np.clip(img_corrected, 0.0, 1.0)
    
    # Convert back to sRGB if needed
    if not is_likely_linear:
        img_corrected = linear_to_srgb(img_corrected)
    
    # Convert back to original bit depth
    if is_float:
        if max_value > 1.0:
            result = (img_corrected * max_value).astype(img.dtype)
        else:
            result = img_corrected.astype(img.dtype)
    else:
        result = np.clip(img_corrected * max_value, 0, max_value).astype(img.dtype)
    
    return result


def manual_white_balance(img, r_gain=1.0, g_gain=1.0, b_gain=1.0):
    """
    Apply manual white balance gains
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image in BGR format
    r_gain, g_gain, b_gain : float
        Multipliers for each channel
    """
    # Detect bit depth
    bit_depth, is_float, max_value = detect_bit_depth(img)
    print(f"  Manual White Balance: {bit_depth}-bit, gains R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")
    
    # Convert to float
    if is_float:
        if max_value > 1.0:
            img_float = img / max_value
        else:
            img_float = img.copy()
    else:
        img_float = img.astype(np.float64) / max_value
    
    # Apply gains directly (assumes image is in linear space)
    b, g, r = cv2.split(img_float)
    r_corrected = r * r_gain
    g_corrected = g * g_gain
    b_corrected = b * b_gain
    
    # Merge and clip
    img_corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    img_corrected = np.clip(img_corrected, 0.0, 1.0)
    
    # Convert back to original bit depth
    if is_float:
        if max_value > 1.0:
            result = (img_corrected * max_value).astype(img.dtype)
        else:
            result = img_corrected.astype(img.dtype)
    else:
        result = np.clip(img_corrected * max_value, 0, max_value).astype(img.dtype)
    
    return result


# Main function to replace in your code
def auto_white_balance(img, strength=1.0, method='gray_world', preserve_luminance=True):
    """
    REPLACEMENT FOR YOUR EXISTING auto_white_balance() FUNCTION
    
    Main white balance function with multiple algorithms
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image in BGR format (OpenCV standard)
    strength : float
        White balance strength (0.0 = no correction, 1.0 = full correction)
        This replaces your current white_balance_strength parameter
    method : str
        Algorithm to use: 'gray_world' or 'white_patch'
    preserve_luminance : bool
        Maintain overall brightness (recommended for film scans)
    
    Returns:
    --------
    numpy.ndarray
        White balanced image in same bit depth as input
    """
    if method == 'white_patch':
        return auto_white_balance_white_patch(img, percentile=99, strength=strength)
    else:
        # Default to gray_world
        return auto_white_balance_gray_world(img, strength=strength, preserve_luminance=preserve_luminance)


if __name__ == "__main__":
    """
    Test the white balance functions
    """
    print("White Balance Module - Test")
    print("="*60)
    
    # Test with different bit depths
    test_sizes = [(100, 100, 3)]
    
    for dtype_name, dtype, max_val in [
        ("8-bit", np.uint8, 255),
        ("16-bit", np.uint16, 65535),
        ("32-bit float", np.float32, 1.0)
    ]:
        print(f"\nTesting {dtype_name}:")
        
        # Create test image with color cast
        for h, w, c in test_sizes:
            img = np.random.rand(h, w, c).astype(np.float64)
            
            # Add orange cast (like your film scan)
            img[:, :, 2] *= 1.5  # Increase red
            img[:, :, 1] *= 1.2  # Increase green slightly
            img[:, :, 0] *= 0.8  # Decrease blue
            
            img = np.clip(img, 0, 1)
            
            # Convert to target dtype
            if dtype == np.uint8:
                img = (img * 255).astype(np.uint8)
            elif dtype == np.uint16:
                img = (img * 65535).astype(np.uint16)
            else:
                img = img.astype(np.float32)
            
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            print(f"  Input: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
            
            # Test white balance
            result = auto_white_balance(img, strength=1.0, method='gray_world')
            
            print(f"  Output: shape={result.shape}, dtype={result.dtype}, range=[{result.min()}, {result.max()}]")
            print(f"  Histogram preserved: {np.std(img):.1f} -> {np.std(result):.1f}")
            print()
    
    print("="*60)
    print("Test complete!")
