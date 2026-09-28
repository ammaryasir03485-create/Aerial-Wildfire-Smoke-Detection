from ultralytics import YOLO

def main():
    # 1. Apne trained model ke weights load karein
    model = YOLO("runs/detect/aerial_fire_fast/weights/best.pt")

    # 2. Test images ya video par inference run karein
    # Aap yahan kisi test folder ka path ya kisi aik image/video ka path de sakte hain
    results = model.predict(
        source="C:/Users/NICAT/Desktop/fire_detection cv/test_images",  # Aap yahan koi aik image ( jaise "test.jpg" ) ya video bhi de sakte hain
        conf=0.5,            # Confidence threshold (50% se upar wale detections dikhayega)
        show=True,           # Screen par bounding boxes wali window show karega
        save=True            # Results ko save bhi kar lega
    )

    print("Testing mukammal ho gayi hai! Saved results 'runs/detect/predict' folder mein mil jayenge.")

if __name__ == '__main__':
    main()