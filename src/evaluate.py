import os
import pandas as pd
import numpy as np
from ultralytics import YOLO


# Settings
VAL_IMAGE_DIR = 'dataset/val/images'
CSV_PATH = 'data/annotations.csv'
MODEL_PATH = 'runs/obb/train/weights/best.pt'
DIST_THRESHOLD = 5.0  # Match if center is within 5 pixels
DEBUG_PRINTS = False


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model missing at {MODEL_PATH}. Train the model first.")
        return

    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded")

    df = pd.read_csv(CSV_PATH)
    val_images = os.listdir(VAL_IMAGE_DIR)

    total_gt_tubes = 0
    total_pred_tubes = 0
    true_positives = 0
    angle_errors = []

    print(f"Evaluating model against {len(val_images)} validation images...")

    for img_name in val_images:
        img_path = os.path.join(VAL_IMAGE_DIR, img_name)
        img_gt = df[df['image'] == img_name]
        total_gt_tubes += len(img_gt)

        # Run inference
        results = model(img_path, verbose=False)
        preds = []

        for result in results:
            if result.obb is not None:
                boxes = result.obb.xywhr.cpu().numpy()
                confs = result.obb.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    if conf > 0.30:  # Filter background noise
                        cx, cy, _, _, rad = box
                        deg = (-np.degrees(rad)) % 360
                        preds.append({'x': cx, 'y': cy, 'angle': deg})

        total_pred_tubes += len(preds)

        # Distance-based evaluation
        matched_preds = set()
        for _, gt_row in img_gt.iterrows():
            gt_cx, gt_cy = gt_row['center_x'], gt_row['center_y']
            gt_angle = gt_row['angle_deg']

            best_match_idx = -1
            min_dist = float('inf')

            for idx, p in enumerate(preds):
                if idx in matched_preds:
                    continue
                dist = np.sqrt((gt_cx - p['x'])**2 + (gt_cy - p['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx

            if min_dist <= DIST_THRESHOLD:
                true_positives += 1
                matched_preds.add(best_match_idx)

                # Wrapped Angular Error Calculation
                pred_angle = preds[best_match_idx]['angle']
                raw_diff = (pred_angle - gt_angle) % 360
                wrapped360 = min(raw_diff, 360 - raw_diff)
                # For oriented boxes, 180-periodicity may be relevant
                raw180 = raw_diff % 180
                wrapped180 = min(raw180, 180 - raw180)
                # Check 90-degree flips
                raw90 = raw_diff % 90
                wrapped90 = min(raw90, 90 - raw90)

                # Use 180-degree periodicity but allow a 90-degree flip
                # (handles cases where width/height get swapped)
                ang_err = min(wrapped180, wrapped90)
                angle_errors.append(ang_err)

                if DEBUG_PRINTS and len(angle_errors) <= 20:
                    dbg_msg = (
                        f"DBG: img={img_name} gt={gt_angle:.1f}"
                        f" pred={pred_angle:.1f} raw={raw_diff:.1f}"
                        f" w360={wrapped360:.1f}"
                        f" w180={wrapped180:.1f} w90={wrapped90:.1f}"
                    )
                    print(dbg_msg)

    # Compile Final Metrics
    precision = (true_positives / total_pred_tubes
                 if total_pred_tubes > 0 else 0)
    recall = (true_positives / total_gt_tubes
              if total_gt_tubes > 0 else 0)
    f1 = (2 * (precision * recall) / (precision + recall)
          if (precision + recall) > 0 else 0)
    mean_angle_error = np.mean(angle_errors) if angle_errors else 0

    print("\nEvaluation summary")
    print(f"Total ground truth tubes: {total_gt_tubes}")
    print(f"Total detections: {total_pred_tubes}")
    print(f"True positives: {true_positives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Mean absolute angle error: {mean_angle_error:.2f} degrees")


if __name__ == '__main__':
    main()
