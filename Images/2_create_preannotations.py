import base64
import json
from pathlib import Path

import cv2

base_dir = Path(__file__).resolve().parent
images_dir = base_dir / "images_1"
masks_dir = base_dir / "generated_masks"

image_files = (
    list(images_dir.glob("*.png"))
    + list(images_dir.glob("*.jpg"))
    + list(images_dir.glob("*.jpeg"))
)

tasks = []

for img_path in image_files:
    # Match the mask filename generated in Step 1
    mask_path = masks_dir / f"{img_path.stem}_mask.png"

    if not mask_path.exists():
        print(f"[SKIP] Mask not found for {img_path.name}")
        continue

    # 1. Read mask and image dimensions
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape

    # 2. Extract bubble contour polygons from the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    results = []
    for i, cnt in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        # Convert pixel points to percentage coordinates (required by Label Studio)
        points = []
        for pt in approx:
            x_pct = float(pt[0][0]) / w * 100.0
            y_pct = float(pt[0][1]) / h * 100.0
            points.append([x_pct, y_pct])

        results.append(
            {
                "id": f"bubble_{i}",
                "type": "polygonlabels",
                "from_name": "poly_labels",
                "to_name": "image",
                "original_width": w,
                "original_height": h,
                "image_rotation": 0,
                "value": {"points": points, "polygonlabels": ["Bubble"]},
            }
        )

    # 3. Base64-encode raw image to embed it directly into the JSON
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    tasks.append(
        {
            "data": {"image": f"data:image/png;base64,{img_b64}"},
            "predictions": [{"model_version": "opencv_v1", "result": results}],
        }
    )

output_json = base_dir / "preannotations.json"
with open(output_json, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"\n[DONE] Successfully created {output_json.name} with {len(tasks)} tasks!")
