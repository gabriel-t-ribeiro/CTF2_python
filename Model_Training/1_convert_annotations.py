import json
from pathlib import Path

import cv2
import numpy as np

# Directory Configuration
BASE_DIR = Path(__file__).resolve().parent
EXPORT_JSON = BASE_DIR / "project_export.json"  # Exported from Label Studio
OUTPUT_IMG_DIR = BASE_DIR / "dataset" / "images"
OUTPUT_MASK_DIR = BASE_DIR / "dataset" / "masks"

OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


def convert_labelstudio_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    converted_count = 0

    for task in tasks:
        # Ignore unannotated tasks
        if not task.get("annotations") or len(task["annotations"]) == 0:
            continue

        annotation_results = task["annotations"][0].get("result", [])
        if not annotation_results:
            continue

        # Extract canvas dimensions
        w = annotation_results[0]["original_width"]
        h = annotation_results[0]["original_height"]
        task_id = task["id"]

        # 0 = liquid background, 255 = gas bubbles
        mask = np.zeros((h, w), dtype=np.uint8)

        for item in annotation_results:
            if item.get("type") == "polygonlabels":
                points = item["value"]["points"]
                # Convert percentage-based points to absolute pixel coordinates
                pts_px = np.array(
                    [[int(pt[0] * w / 100.0), int(pt[1] * h / 100.0)] for pt in points],
                    dtype=np.int32,
                )
                # Fill polygon (logical OR on overlapping bubbles)
                cv2.fillPoly(mask, [pts_px], 255)

        mask_file = OUTPUT_MASK_DIR / f"frame_{task_id:04d}.png"
        cv2.imwrite(str(mask_file), mask)
        converted_count += 1

    print(f"Successfully converted {converted_count} masks to {OUTPUT_MASK_DIR}")


if __name__ == "__main__":
    convert_labelstudio_json(EXPORT_JSON)
