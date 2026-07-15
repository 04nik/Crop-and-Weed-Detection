# prepare_dataset.py
import os
import shutil
import random
import argparse
import glob

def setup_yolo_directories(output_dir):
    """Create directory structure required for YOLOv8 training."""
    subdirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    print(f"Created YOLO dataset directories in: {output_dir}")

def generate_yaml(output_dir, nc=2, class_names=["sesame", "weed"]):
    """Generate the data.yaml file required for YOLO training."""
    # Use absolute path to avoid directory resolution errors in Ultralytics
    abs_output_path = os.path.abspath(output_dir).replace('\\', '/')
    
    yaml_content = f"""path: {abs_output_path}
train: images/train
val: images/val
test: images/test

nc: {nc}
names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"
        
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Generated YAML config: {yaml_path}")

def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.15):
    # Find all images (supporting png, jpg, jpeg)
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
        
    if not image_files:
        print(f"No images found in {input_dir}. Please make sure images are in this directory.")
        return
        
    print(f"Found {len(image_files)} images. Finding matching YOLO label text files...")
    
    paired_data = []
    missing_labels = 0
    
    for img_path in image_files:
        # Check for matching label txt file
        base_name = os.path.splitext(img_path)[0]
        # In case labels are in a separate folder, check both same directory and look in sibling directories
        label_path = base_name + ".txt"
        
        # If not found directly, check if we have a sibling 'labels' directory
        if not os.path.exists(label_path):
            parts = img_path.split(os.sep)
            if "images" in parts:
                # Replace last occurrences of 'images' with 'labels'
                idx = len(parts) - 1 - parts[::-1].index("images")
                parts[idx] = "labels"
                # Change extension of the filename part
                filename = parts[-1]
                parts[-1] = os.path.splitext(filename)[0] + ".txt"
                alt_label_path = os.sep.join(parts)
                if os.path.exists(alt_label_path):
                    label_path = alt_label_path
                    
        if os.path.exists(label_path):
            paired_data.append((img_path, label_path))
        else:
            missing_labels += 1
            # Add with none, but we might skip during copy if labels are required
            paired_data.append((img_path, None))
            
    print(f"Paired data: {len(paired_data) - missing_labels} items matched. {missing_labels} images missing labels.")
    
    # Shuffle and split
    random.shuffle(paired_data)
    
    total = len(paired_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        "train": paired_data[:train_end],
        "val": paired_data[train_end:val_end],
        "test": paired_data[val_end:]
    }
    
    setup_yolo_directories(output_dir)
    
    for split_name, items in splits.items():
        print(f"Copying {len(items)} items to '{split_name}' split...")
        copied_count = 0
        for img_path, label_path in items:
            # We skip files without labels to prevent training errors in YOLO
            if label_path is None:
                continue
                
            dest_img_path = os.path.join(output_dir, f"images/{split_name}", os.path.basename(img_path))
            dest_lbl_path = os.path.join(output_dir, f"labels/{split_name}", os.path.basename(label_path))
            
            shutil.copy(img_path, dest_img_path)
            shutil.copy(label_path, dest_lbl_path)
            copied_count += 1
            
        print(f"Successfully copied {copied_count} paired files to '{split_name}'")
        
    generate_yaml(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splits raw dataset folders into YOLO Train/Val/Test formats.")
    parser.add_argument("--input", type=str, default="./dataset", help="Input directory containing raw images and labels")
    parser.add_argument("--output", type=str, default="./yolo_dataset", help="Output directory for split YOLO dataset")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio (0-1)")
    parser.add_argument("--val", type=float, default=0.15, help="Val ratio (0-1)")
    
    args = parser.parse_args()
    
    split_dataset(args.input, args.output, args.train, args.val)
