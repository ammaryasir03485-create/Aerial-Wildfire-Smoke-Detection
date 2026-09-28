from ultralytics import YOLO

def main():
    # 1. Load the lightweight YOLOv8 nano model for fast training
    model = YOLO("yolov8n.pt")

    # 2. Start training with your dataset configured via data.yaml
    results = model.train(
        data="data.yaml",          # Same folder mein hone ki wajah se direct naam likhna kafi hai
        epochs=50,                 # 50 epochs model training ke liye
        imgsz=640,                 # Standard image size for fast training & memory efficiency
        batch=16,                  # Batch size
        name="aerial_fire_fast"    # Output folder name
    )

    print("Training mukammal ho gayi hai!")

if __name__ == '__main__':
    main()