from ultralytics import YOLO


def main():
    # Load the OBB specific model
    print("Loading YOLO model")
    model = YOLO('yolov8n-obb.pt')
    print("Model loaded successfully")

    print("Starting training")
    model.train(
        data='dataset/data.yaml',
        epochs=60,         # 60 is good for 70 images
        imgsz=640,
        batch=4,
        workers=2,
        device='cpu',      # Explicitly use CPU if no GPU available
        degrees=360.0,     # Essential for tubes
        # Augmentations adjusted for 0-360 detection:
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        scale=0.3,
        translate=0.1,
        fliplr=0.0,        # Set to 0 to avoid mirroring the angles
        mosaic=1.0         # Highly recommended for small datasets
    )
    print("Training complete")


if __name__ == '__main__':
    main()
