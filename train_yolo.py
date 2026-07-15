# train_yolo.py
import argparse
import os
import sys

def check_dependencies():
    """Verify that required modules are installed."""
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("Error: PyTorch is not installed. Please install PyTorch before running training.")
        sys.exit(1)
        
    try:
        import ultralytics
        from ultralytics import YOLO
        print(f"Ultralytics Version: {ultralytics.__version__}")
    except ImportError:
        print("Error: Ultralytics package is not installed. Installing it via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
        print("Ultralytics installed successfully!")

def run_training(data_yaml, epochs=50, batch_size=-1, imgsz=512):
    check_dependencies()
    from ultralytics import YOLO
    import torch
    
    # 1. Choose training device
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on device: {device}")
    
    # 2. Check if data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"Error: YAML configuration file {data_yaml} does not exist.")
        print("Please run prepare_dataset.py first to partition the dataset and generate data.yaml.")
        sys.exit(1)
        
    # 3. Load pre-trained YOLOv8 Nano weights (fastest and lightweight)
    print("Loading pre-trained YOLOv8n backbone model...")
    model = YOLO("yolov8n.pt")
    
    # 4. Run training
    print(f"Training on custom dataset '{data_yaml}' for {epochs} epochs...")
    
    # Batch size of -1 indicates auto-batching to fit GPU/CPU RAM limits
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=4,
        save=True,
        project="sesame_weed_yolo",
        name="train"
    )
    
    print("\nTraining Completed!")
    best_weights = os.path.join("sesame_weed_yolo", "train", "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"Best model weights saved to: {best_weights}")
    else:
        print("Training ran successfully, check 'sesame_weed_yolo/' folder for logs and weights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 model on sesame crop and weeds.")
    parser.add_argument("--data", type=str, default="./yolo_dataset/data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto-batching)")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image dimensions (square)")
    
    args = parser.parse_args()
    
    run_training(args.data, args.epochs, args.batch, args.imgsz)
