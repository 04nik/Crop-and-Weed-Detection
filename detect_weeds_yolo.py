# detect_weeds_yolo.py
import cv2
import numpy as np
import json
import argparse
import os
import sys

def detect_yolo_spray(image_path, weights_path="sesame_weed_yolo/train/weights/best.pt", output_path="test_data/field_result_yolo.jpg", num_nozzles=5):
    # Verify Ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: Ultralytics package is not installed. Run 'pip install ultralytics'.")
        return None
        
    # Check input image
    if not os.path.exists(image_path):
        print(f"Error: Input image {image_path} not found.")
        return None
        
    # Check weights path, fallback to pre-trained yolov8n if needed
    if not os.path.exists(weights_path):
        print(f"Warning: Custom weights '{weights_path}' not found.")
        fallback_weights = "yolov8n.pt"
        print(f"Falling back to pre-trained model '{fallback_weights}' for structural pipeline verification.")
        weights_path = fallback_weights
        
    # 1. Load YOLO model
    print(f"Loading YOLOv8 model weights from: {weights_path}")
    model = YOLO(weights_path)
    
    # 2. Run inference
    print(f"Running inference on: {image_path}")
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    
    # Run prediction
    results = model(img, imgsz=512)
    predictions = results[0]
    
    # 3. Setup Sprayer nozzle mapping
    nozzle_width = width / num_nozzles
    nozzle_triggers = [[] for _ in range(num_nozzles)]
    
    annotated_img = img.copy()
    
    # Draw nozzle boundaries on overlay
    for i in range(1, num_nozzles):
        nx = int(i * nozzle_width)
        cv2.line(annotated_img, (nx, 0), (nx, height), (255, 150, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(annotated_img, f"Nozzle {i}", (nx - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
    cv2.putText(annotated_img, f"Nozzle {num_nozzles}", (width - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)

    crops_count = 0
    weeds_count = 0
    
    # Parse YOLO boxes
    if hasattr(predictions, 'boxes') and predictions.boxes is not None:
        boxes = predictions.boxes
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Class ID and confidence
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Map classes:
            # Custom Model expectation: 0: sesame (crop), 1: weed (weed)
            # Default pre-trained coco model fallback: we'll treat 'potted plant' (id 58) or other plant classes as crop, and others as weed.
            is_crop = True
            
            if weights_path == "yolov8n.pt":
                # COCO classes fallback mapping
                # COCO Class 58 is 'potted plant', Class 64 is 'pottedplant' alternative, Class 0 is person, etc.
                # Treat plant classes as crop, everything else as weed for testing
                if cls_id in [58, 64]:
                    is_crop = True
                else:
                    is_crop = False
            else:
                # Custom Weights mapping
                if cls_id == 0:
                    is_crop = True
                else:
                    is_crop = False
            
            # Calculate centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            if is_crop:
                crops_count += 1
                # Draw Crop (Green Box)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, f"Sesame ({conf:.2f})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                weeds_count += 1
                # Draw Weed (Red Box)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_img, f"Weed ({conf:.2f})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Map centroid horizontal position to active nozzle channel
                nozzle_idx = int(cx // nozzle_width)
                if 0 <= nozzle_idx < num_nozzles:
                    # Pad boundaries for sprayer spray coverage zone
                    y_start = max(0, y1 - 15)
                    y_end = min(height, y2 + 15)
                    nozzle_triggers[nozzle_idx].append({"y_start": y_start, "y_end": y_end})
                    
                    # Highlight target zone on nozzle strip
                    cv2.rectangle(annotated_img, (int(nozzle_idx * nozzle_width) + 2, y_start), (int((nozzle_idx + 1) * nozzle_width) - 2, y_end), (255, 255, 0), 1)
                    
    # 4. Calculate pesticide saved metrics
    spray_mask = np.zeros((height, width), dtype=np.uint8)
    for idx, triggers in enumerate(nozzle_triggers):
        n_x_start = int(idx * nozzle_width)
        n_x_end = int((idx + 1) * nozzle_width)
        for t in triggers:
            spray_mask[t["y_start"]:t["y_end"], n_x_start:n_x_end] = 255
            
    sprayed_pixels = np.sum(spray_mask == 255)
    total_pixels = width * height
    pesticide_saved_ratio = 1.0 - (sprayed_pixels / total_pixels) if total_pixels > 0 else 1.0
    pesticide_saved_percent = round(pesticide_saved_ratio * 100, 2)
    
    # 5. Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, annotated_img)
    
    stats = {
        "image_size": [width, height],
        "crops_detected": crops_count,
        "weeds_detected": weeds_count,
        "sprayed_area_px": int(sprayed_pixels),
        "total_area_px": total_pixels,
        "pesticide_saved_percent": pesticide_saved_percent,
        "nozzle_triggers": nozzle_triggers
    }
    
    json_path = os.path.splitext(output_path)[0] + "_data.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nYOLO Processing Complete:")
    print(f"  Crops (Sesame) detected: {crops_count}")
    print(f"  Weeds detected: {weeds_count}")
    print(f"  Pesticide Saved: {pesticide_saved_percent}%")
    print(f"  Result saved to: {output_path}")
    print(f"  Data JSON saved to: {json_path}")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate field image using custom YOLO model and map sprayer triggers.")
    parser.add_argument("--input", type=str, default="test_data/field_sample.jpg", help="Path to input field image")
    parser.add_argument("--weights", type=str, default="sesame_weed_yolo/train/weights/best.pt", help="Path to custom trained weights best.pt")
    parser.add_argument("--output", type=str, default="test_data/field_result_yolo.jpg", help="Path to write output image")
    parser.add_argument("--nozzles", type=int, default=5, help="Number of spray nozzles across boom")
    
    args = parser.parse_args()
    
    detect_yolo_spray(args.input, args.weights, args.output, args.nozzles)
