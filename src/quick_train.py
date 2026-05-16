from ultralytics import YOLO


def main():
    print("Loading YOLO model")
    model = YOLO('yolov8n-obb.pt')
    print("Model loaded successfully")

    print("Starting quick training for 2 epochs")
    model.train(
        data='dataset/data.yaml',
        epochs=2,
        imgsz=640,
        batch=4,
        workers=2,
        degrees=360.0,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        scale=0.3,
        translate=0.1,
        fliplr=0.0,        # Set to 0 to avoid mirroring the angles
        mosaic=1.0
    )
    print("Quick training complete")


if __name__ == '__main__':
    main()
