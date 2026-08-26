import argparse
import base64
import json
from pathlib import Path
import cv2
import numpy as np


def convert_annotations_and_extract_images(json_path: Path, dataset_dir: Path):
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    count = 0
    for task in tasks:
        if not task.get("annotations") or len(task["annotations"]) == 0:
            continue

        annotation_results = task["annotations"][0].get("result", [])
        if not annotation_results:
            continue

        w = annotation_results[0]["original_width"]
        h = annotation_results[0]["original_height"]
        task_id = task["id"]
        filename = f"frame_{task_id:04d}.png"

        # 1. Extract and Save the Raw Image from Base64
        image_data = task["data"].get("image", "")
        if "base64," in image_data:
            b64_str = image_data.split("base64,")[1]
            img_bytes = base64.b64decode(b64_str)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            raw_img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(img_dir / filename), raw_img)

        # 2. Rasterize and Save the Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for item in annotation_results:
            if item.get("type") == "polygonlabels":
                points = item["value"]["points"]
                pts_px = np.array(
                    [[int(pt[0] * w / 100.0), int(pt[1] * h / 100.0)] for pt in points],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [pts_px], 255)

        cv2.imwrite(str(mask_dir / filename), mask)
        count += 1

    print(f"[DONE] Extracted {count} images to: {img_dir}")
    print(f"[DONE] Extracted {count} masks to: {mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        "-j",
        type=Path,
        required=True,
        help="Path to Label Studio JSON",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=Path("dataset"),
        help="Root dataset directory",
    )
    args = parser.parse_args()

    convert_annotations_and_extract_images(args.json, args.dataset)
