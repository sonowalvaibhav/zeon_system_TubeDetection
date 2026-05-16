import os
import pandas as pd
import numpy as np
import cv2
import shutil

# Paths
CSV_PATH = 'data/annotations.csv'
IMG_DIR = 'data/annotated_images'
OUTPUT_DIR = 'dataset'


def get_obb_corners(cx, cy, w, h, angle_deg):
    """Convert box center, extents, and angle to 4 corner coordinates."""
    # Convert counter-clockwise angle to radians
    # Negate because image Y increases downward
    theta = np.radians(-angle_deg)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    # Half extents
    dx = w / 2.0
    dy = h / 2.0

    # Coordinate system rotation matrix offsets
    corners = [
        (-dx, -dy),  # Top-Left
        (dx, -dy),   # Top-Right
        (dx, dy),    # Bottom-Right
        (-dx, dy)    # Bottom-Left
    ]

    rotated_corners = []
    for x, y in corners:
        rx = cx + (x * cos_a - y * sin_a)
        ry = cy + (x * sin_a + y * cos_a)
        rotated_corners.append((rx, ry))

    return rotated_corners


def main():
    print(f"Reading annotations from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    images = df['image'].unique()
    print(f"Found {len(images)} unique images")

    # Shuffle and split 80% train, 20% validation
    np.random.seed(42)
    np.random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    print(f"Splitting: {len(train_imgs)} train, {len(val_imgs)} val")
    splits = {'train': train_imgs, 'val': val_imgs}

    for split_name, img_list in splits.items():
        img_out = os.path.join(OUTPUT_DIR, split_name, 'images')
        lbl_out = os.path.join(OUTPUT_DIR, split_name, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        print(f"\nProcessing {split_name} ({len(img_list)} images)")

        for img_name in img_list:
            src_img_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(src_img_path):
                print(f"Missing source image: {img_name}")
                continue

            # Copy image
            shutil.copy(src_img_path, os.path.join(img_out, img_name))

            # Read metadata to normalize dimensions
            img = cv2.imread(src_img_path)
            if img is None:
                print(f"Failed to read: {img_name}")
                continue
            h_img, w_img, _ = img.shape

            # Process labels
            img_df = df[df['image'] == img_name]
            label_file = os.path.join(lbl_out,
                                      img_name.replace('.png', '.txt'))

            with open(label_file, 'w') as f:
                for _, row in img_df.iterrows():
                    corners = get_obb_corners(
                        row['center_x'], row['center_y'],
                        row['bbox_w'], row['bbox_h'],
                        row['angle_deg'])

                    # Normalize points between 0 and 1
                    flat_corners = []
                    for x, y in corners:
                        flat_corners.extend([x / w_img, y / h_img])

                    corner_str = " ".join(
                        [f"{coord:.6f}" for coord in flat_corners])
                    # 0 represents class index 'tube_lid'
                    f.write(f"0 {corner_str}\n")

        print(f"{split_name} processing complete")


if __name__ == '__main__':
    main()
    print("Data conversion and splitting complete")
