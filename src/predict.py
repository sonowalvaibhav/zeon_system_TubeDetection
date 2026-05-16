from ultralytics import YOLO

MODEL_PATH = 'runs/obb/train/weights/best.pt'
IMAGE_URL = (
    'https://media.istockphoto.com/id/1334661211/photo/'
    'overhead-shot-of-researchers-hand-placing-tube-in-'
    'laboratory-centrifuge.jpg'
    '?s=1024x1024&w=is&k=20&c=QHmq5_71QAEa3H7-'
    'hBUm1E71VTVX5ZI1Z2hY_FZHJ98='
)


def main():
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=IMAGE_URL,
        save=True,
        conf=0.25,
        imgsz=640,
    )

    for result in results:
        if result.obb is not None:
            for box in result.obb.xywhr:
                x, y, w, h, r = box.tolist()
                angle_deg = (-float(r) * 180 / 3.14159) % 360
                print(
                    f"Tube detected at ({x:.1f}, {y:.1f}) "
                    f"with angle {angle_deg:.2f} degrees"
                )


if __name__ == '__main__':
    main()
