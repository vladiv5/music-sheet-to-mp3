import cv2
import numpy as np
import pytesseract

def detect_octave_shifts(image_path, staffs, interline):
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    import time
    t0 = time.time()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optimization: Resize for OCR if image is too large. 
    # OCR doesn't need 3000px+ resolution to find '8va'.
    h, w = gray.shape
    scale = 1.0
    if h > 1500:
        scale = 1500.0 / h
        gray_small = cv2.resize(gray, (int(w * scale), 1500))
    else:
        gray_small = gray
        
    _, bin_img = cv2.threshold(gray_small, 127, 255, cv2.THRESH_BINARY)
    
    # Run OCR with PSM 6 (Assume a single uniform block of text)
    custom_config = r'--psm 6'
    data = pytesseract.image_to_data(bin_img, output_type=pytesseract.Output.DICT, config=custom_config)
    
    print(f"[OctaveDetector] ⏱️ Tesseract OCR took {time.time() - t0:.2f}s (scale={scale:.2f})")
    
    shifts = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip().lower()
        if text in ['8va', 'gva', 'sva']:
            amount = 1
        elif text in ['8vb', 'gvb', 'svb']:
            amount = -1
        else:
            continue
            
        # Rescale coordinates back to original size
        x, y, w, h = [int(v / scale) for v in (data['left'][i], data['top'][i], data['width'][i], data['height'][i])]
        
        # Find closest staff. 8va is usually above, 8vb is below.
        closest_staff = min(staffs, key=lambda s: abs(s.top - y) if amount == 1 else abs(s.bottom - y))
        
        # ROI for dotted line. Restrict vertically to text height to avoid staff lines.
        roi_y1 = max(0, y - 5)
        roi_y2 = min(img.shape[0], y + h + 5)
        roi_x1 = x + w
        roi_x2 = img.shape[1]
        
        x_end = roi_x1 + int(interline * 10) # default fallback
        
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            _, roi_bin = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Dilate horizontally to connect the dots/dashes
            kernel = np.ones((1, int(interline * 3)), np.uint8)
            dilated = cv2.dilate(roi_bin, kernel, iterations=1)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                # The dotted line should start relatively close to the text and be thin
                if cx < interline * 5 and ch < interline * 2:
                    valid_contours.append((cx, cy, cw, ch))
            
            if valid_contours:
                # Pick the longest valid contour
                best = max(valid_contours, key=lambda b: b[2])
                x_end = roi_x1 + best[0] + best[2]
                
        shifts.append({
            'staff_id': id(closest_staff),
            'x_start': x,
            'x_end': x_end,
            'amount': amount
        })
        
    return shifts
