"""
core/volta_detector.py
----------------------
Detects volta brackets (ending 1, ending 2) using OCR and line detection.
Used to identify repetition endings in musical scores.
"""
import cv2
import numpy as np

def detect_voltas(image_path, staffs):
    """
    Scans the image for volta brackets (endings) using OCR for '1.' and '2.'.
    Returns a list of volta objects with horizontal extent and ending number.
    """
    try:
        import pytesseract
    except ImportError:
        print("[VoltaDetector] ⚠️ pytesseract not available, skipping volta detection")
        return []

    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Resize for OCR optimization
    scale = 1.0
    if h > 1500:
        scale = 1500.0 / h
        gray_small = cv2.resize(gray, (int(w * scale), 1500))
    else:
        gray_small = gray

    # Run OCR
    custom_config = r'--psm 6'
    data = pytesseract.image_to_data(gray_small, output_type=pytesseract.Output.DICT, config=custom_config)
    
    voltas = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        
        # Look for "1.", "1", "2.", "2"
        ending_num = None
        if text in ['1.', '1']:
            ending_num = 1
        elif text in ['2.', '2']:
            ending_num = 2
            
        if ending_num:
            x, y, bw, bh = [int(v / scale) for v in (data['left'][i], data['top'][i], data['width'][i], data['height'][i])]
            
            # Find closest staff (volta is usually above the staff)
            closest_staff = min(staffs, key=lambda s: abs(s.top - y))
            
            # Check if it's actually above the staff (within reasonable distance)
            if abs(closest_staff.top - y) > 150:
                continue
                
            # Scan for horizontal line to the right of the text
            # ROI: from text to the right
            roi_x1 = x
            roi_x2 = min(img.shape[1], x + 400) # Volta brackets are usually long
            roi_y1 = max(0, y - 10)
            roi_y2 = min(img.shape[0], y + bh + 10)
            
            x_end = x + 100 # Default fallback
            
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
                _, roi_bin = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
                
                # Use horizontal kernel to find the top bar of the volta
                kernel = np.ones((1, 40), np.uint8)
                dilated = cv2.dilate(roi_bin, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the widest horizontal contour
                    best_c = max(contours, key=lambda c: cv2.boundingRect(c)[2])
                    cx, cy, cw, ch = cv2.boundingRect(best_c)
                    if cw > 30: # At least 30 pixels long
                        x_end = roi_x1 + cx + cw

            voltas.append({
                'staff_id': id(closest_staff),
                'x_start': x,
                'x_end': x_end,
                'number': ending_num
            })

    if voltas:
        print(f"[VoltaDetector] 🎯 Found {len(voltas)} volta bracket(s).")
    
    return voltas
