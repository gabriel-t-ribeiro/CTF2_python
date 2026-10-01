#!/usr/bin/env python3
import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    return "".join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def decode_rle_brush(rle, height: int, width: int) -> np.ndarray:
    """Decodes Label Studio 4-channel RGBA RLE brush bitstream into 2D uint8 mask."""
    inp = InputStream(bytes2bit(rle))
    num = inp.read(32)
    word_size = inp.read(5) + 1
    rle_sizes = [inp.read(4) + 1 for _ in range(4)]
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = inp.read(1)
        j = i + 1 + inp.read(rle_sizes[inp.read(2)])
        if x:
            val = inp.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                out[i] = inp.read(word_size)
                i += 1

    if num == height * width * 4:
        rgba = out.reshape((height, width, 4))
        # Mask is positive where alpha > 0 (or any non-zero channel)
        mask = (rgba[:, :, 3] > 0).astype(np.uint8) * 255
    elif num == height * width:
        mask = (out.reshape((height, width)) > 0).astype(np.uint8) * 255
    else:
        mask = (out[: height * width].reshape((height, width)) > 0).astype(np.uint8) * 255
    return mask


def convert_annotations_and_extract_images(json_path: Path, dataset_dir: Path):
    import time

    start_time = time.perf_counter()

    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
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

        # 2. Rasterize and Save the Mask (polygons + RLE brushes)
        mask = np.zeros((h, w), dtype=np.uint8)
        for item in annotation_results:
            item_type = item.get("type")
            if item_type == "polygonlabels":
                points = item["value"]["points"]
                pts_px = np.array(
                    [[round(pt[0] * w / 100.0), round(pt[1] * h / 100.0)] for pt in points],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [pts_px], 255)
            elif item_type == "brushlabels":
                rle = item["value"]["rle"]
                brush_mask = decode_rle_brush(rle, h, w)
                mask = np.maximum(mask, brush_mask)

        cv2.imwrite(str(mask_dir / filename), mask)
        count += 1

    elapsed = time.perf_counter() - start_time
    print(f"[DONE] Extracted {count} images to: {img_dir}")
    print(f"[DONE] Extracted {count} masks to: {mask_dir}")
    print(
        f"[TIMING] Dataset extraction completed in {elapsed:.3f} seconds ({elapsed / max(1, count):.4f} s/frame)"
    )


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
