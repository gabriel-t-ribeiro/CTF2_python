#!/usr/bin/env python3
import argparse
import importlib.util
import tempfile
from pathlib import Path


def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Resolve the base directory dynamically
base_dir = Path(__file__).resolve().parent

# Load modules that have numbers in their filenames
batch_inference = load_module_from_file(
    "batch_inference", base_dir / "model_training" / "3_batch_inference.py"
)
generate_masks = load_module_from_file(
    "generate_masks", base_dir / "image_scripts" / "2_generate_masks.py"
)
create_preannotations = load_module_from_file(
    "create_preannotations", base_dir / "image_scripts" / "3_create_preannotations.py"
)
convert_annotations = load_module_from_file(
    "convert_annotations", base_dir / "model_training" / "1_convert_annotations.py"
)
train_u_net = load_module_from_file("train_u_net", base_dir / "model_training" / "2_train_u_net.py")


def do_prepare(args):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_masks_dir = Path(temp_dir) / "masks"
        temp_masks_dir.mkdir()

        if args.use_cv:
            print(f"Running OpenCV thresholding to generate masks in {temp_masks_dir}...")
            generate_masks.generate_masks(args.images_dir, temp_masks_dir)
        else:
            if not args.model:
                raise ValueError("Must specify --model when not using --use-cv")
            print(f"Running U-Net inference to generate masks in {temp_masks_dir}...")
            batch_inference.run_inference(args.images_dir, temp_masks_dir, args.model)

        print(f"Creating preannotations JSON at {args.output_json}...")
        create_preannotations.create_preannotations(
            args.images_dir, temp_masks_dir, args.output_json
        )


def do_train(args):
    print(f"Extracting dataset from {args.export_json} to {args.dataset_dir}...")
    convert_annotations.convert_annotations_and_extract_images(args.export_json, args.dataset_dir)

    print(f"Training U-Net model and saving to {args.model_out}...")
    train_u_net.train(
        dataset_dir=args.dataset_dir,
        output_model=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )


def main():
    parser = argparse.ArgumentParser(description="Human-in-the-loop pipeline orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare subcommand
    prep_parser = subparsers.add_parser(
        "prepare",
        help="Prepare dataset by running inference or OpenCV and creating a Label Studio JSON",
    )
    prep_parser.add_argument(
        "--images-dir", type=Path, required=True, help="Input folder of images"
    )
    prep_parser.add_argument(
        "--model",
        type=Path,
        required=False,
        help="Path to existing U-Net model (required unless --use-cv is set)",
    )
    prep_parser.add_argument(
        "--output-json", type=Path, required=True, help="Output path for Label Studio JSON"
    )
    prep_parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Use traditional OpenCV thresholding instead of U-Net inference",
    )

    # train subcommand
    train_parser = subparsers.add_parser(
        "train", help="Train a new U-Net model from a Label Studio exported JSON"
    )
    train_parser.add_argument(
        "--export-json", type=Path, required=True, help="Path to Label Studio exported JSON"
    )
    train_parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="Directory to extract the dataset"
    )
    train_parser.add_argument(
        "--model-out", type=Path, required=True, help="Path to save the new U-Net model"
    )
    train_parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of DataLoader worker processes"
    )

    args = parser.parse_args()

    if args.command == "prepare":
        do_prepare(args)
    elif args.command == "train":
        do_train(args)


if __name__ == "__main__":
    main()
