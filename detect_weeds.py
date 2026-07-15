# detect_weeds.py
import cv2
import numpy as np
import json
import argparse
import os

def detect_vegetation_and_spray(image_path, output_path="result.jpg", num_nozzles=5, crop_row_tolerance=60):
    # 1. Read input image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} does not exist.")
        return None
    
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    
    # 2. Convert to HSV color space for robust color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for green vegetation
    # Hue: 30-90 (covers bright green and yellowish-green leaves)
    # Saturation: 40-255 (excludes brownish soil and stones)
    # Value: 40-255 (excludes very dark shadows and very bright highlights)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # 3. Create green mask and clean up noise with morphology
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphological opening (remove small noise points)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Morphological closing (fill gaps within leaves)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_close)
    
    # 4. Find contours of segmented plant clumps
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pre-scan contours to identify crop rows (vertical lanes where large plants align)
    # We look for large contours and cluster their X coordinates to identify crop rows automatically
    plant_centers_x = []
    large_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800: # Large plants
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                plant_centers_x.append(cx)
                large_contours.append(cnt)
                
    # Detect crop rows by grouping crop center X-coordinates
    # Typically we expect 3 or 4 crop rows in our field views
    crop_rows = []
    if len(plant_centers_x) > 0:
        # Simple 1D clustering of X coordinates
        # Sort X coordinates and group close values together
        sorted_x = sorted(plant_centers_x)
        groups = [[sorted_x[0]]]
        for x in sorted_x[1:]:
            if x - groups[-1][-1] < 100: # Within 100 pixels, same row
                groups[-1].append(x)
            else:
                groups.append([x])
        crop_rows = [int(np.mean(g)) for g in groups]
    else:
        # Fallback to default row spacing if no large plants found
        crop_rows = [width // 4, width // 2, 3 * width // 4]
        
    print(f"Auto-detected crop rows at X coordinates: {crop_rows}")
    
    # 5. Classify each plant contour as Crop or Weed
    crops = []
    weeds = []
    
    # Nozzle zone width calculation
    nozzle_width = width / num_nozzles
    nozzle_triggers = [[] for _ in range(num_nozzles)] # Track spray vertical ranges per nozzle
    
    # We will draw annotations on the output image
    annotated_img = img.copy()
    
    # Draw vertical grid lines for nozzles (blue dashed)
    for i in range(1, num_nozzles):
        nx = int(i * nozzle_width)
        cv2.line(annotated_img, (nx, 0), (nx, height), (255, 150, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(annotated_img, f"Nozzle {i}", (nx - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
    cv2.putText(annotated_img, f"Nozzle {num_nozzles}", (width - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)

    # Draw detected crop rows (green translucent lines)
    overlay = annotated_img.copy()
    for row_x in crop_rows:
        cv2.line(overlay, (row_x, 0), (row_x, height), (0, 255, 0), 6)
    cv2.addWeighted(overlay, 0.2, annotated_img, 0.8, 0, annotated_img)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 30: # Filter out microscopic noise
            continue
            
        # Get bounding box and centroid
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Classification criteria:
        # 1. Close to an auto-detected crop row?
        is_near_crop_row = any(abs(cx - row_x) < crop_row_tolerance for row_x in crop_rows)
        
        # 2. Geometry: crops are larger and more circular/rectangular
        # Weeds are smaller or away from the crop rows
        is_crop = is_near_crop_row and (area > 700)
        
        if is_crop:
            crops.append({"id": i, "bbox": (x, y, w, h), "center": (cx, cy), "area": area})
            # Draw green bounding box for crop
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated_img, "Crop", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            weeds.append({"id": i, "bbox": (x, y, w, h), "center": (cx, cy), "area": area})
            # Draw red bounding box for weed
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(annotated_img, f"Weed ({int(area)}px)", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Map weed to target nozzle
            # Nozzles are indexed 0 to num_nozzles-1 corresponding to their horizontal layout
            nozzle_idx = int(cx // nozzle_width)
            if 0 <= nozzle_idx < num_nozzles:
                # Add spray command for this nozzle: trigger spray from y_start to y_end
                # We add a small padding (e.g. 15 pixels) to make sure we get the entire weed
                y_start = max(0, y - 15)
                y_end = min(height, y + h + 15)
                nozzle_triggers[nozzle_idx].append({"y_start": y_start, "y_end": y_end})
                # Draw spray targets in cyan overlay
                cv2.rectangle(annotated_img, (int(nozzle_idx*nozzle_width)+2, y_start), (int((nozzle_idx+1)*nozzle_width)-2, y_end), (255, 255, 0), 1)

    # 6. Save annotated image
    cv2.imwrite(output_path, annotated_img)
    
    # 7. Calculate savings statistics
    # Total weed bounding box area vs total image area
    weed_sprayed_area = 0
    # Create a blank single-channel mask of the field to measure active spray coverage
    spray_mask = np.zeros((height, width), dtype=np.uint8)
    for idx, triggers in enumerate(nozzle_triggers):
        n_x_start = int(idx * nozzle_width)
        n_x_end = int((idx + 1) * nozzle_width)
        for t in triggers:
            spray_mask[t["y_start"]:t["y_end"], n_x_start:n_x_end] = 255
            
    sprayed_pixels = np.sum(spray_mask == 255)
    total_pixels = width * height
    
    # Pesticide savings ratio
    pesticide_saved_ratio = 1.0 - (sprayed_pixels / total_pixels)
    
    stats = {
        "image_size": [width, height],
        "crop_rows": crop_rows,
        "crops_detected": len(crops),
        "weeds_detected": len(weeds),
        "sprayed_area_px": int(sprayed_pixels),
        "total_area_px": total_pixels,
        "pesticide_saved_percent": round(pesticide_saved_ratio * 100, 2),
        "nozzle_triggers": nozzle_triggers
    }
    
    # Write JSON results
    json_path = os.path.splitext(output_path)[0] + "_data.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nProcessing Complete for: {image_path}")
    print(f"  Crops detected: {len(crops)}")
    print(f"  Weeds detected: {len(weeds)}")
    print(f"  Pesticide Saved: {stats['pesticide_saved_percent']}%")
    print(f"  Result saved to: {output_path}")
    print(f"  Data saved to: {json_path}")
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment crops and weeds and generate nozzle spray triggers.")
    parser.add_argument("--input", type=str, default="test_data/field_sample.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="test_data/field_result.jpg", help="Path to output image")
    parser.add_argument("--nozzles", type=int, default=5, help="Number of spray nozzles across boom")
    parser.add_argument("--tolerance", type=int, default=60, help="Max distance (px) from crop row for crop label")
    
    args = parser.parse_args()
    
    # Ensure test directories exist
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    detect_vegetation_and_spray(args.input, args.output, args.nozzles, args.tolerance)
