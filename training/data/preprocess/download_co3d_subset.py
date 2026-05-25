#!/usr/bin/env python3
"""Download a random CO3Dv2 subset compatible with VGGT's Co3dDataset loader.

The official CO3Dv2 distribution exposes ZIP archives per category chunk, not
individual sequences. This script selects sequences from VGGT annotations,
downloads archives for the selected categories until all selected files are
found, and extracts only the selected images, depths, and depth masks.
"""

import argparse
import gzip
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


CO3D_LINKS_URL = "https://raw.githubusercontent.com/facebookresearch/co3d/main/co3d/links.json"
VGGT_ANNO_URL = "https://huggingface.co/datasets/JianyuanWang/co3d_anno/resolve/main/{name}?download=true"
VGGT_SEEN_CATEGORIES = [
    "apple", "backpack", "banana", "baseballbat", "baseballglove", "bench",
    "bicycle", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "cup", "donut", "hairdryer", "handbag", "hydrant",
    "keyboard", "laptop", "microwave", "motorcycle", "mouse", "orange",
    "parkingmeter", "pizza", "plant", "stopsign", "teddybear", "toaster",
    "toilet", "toybus", "toyplane", "toytrain", "toytruck", "tv", "umbrella",
    "vase", "wineglass",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a random CO3Dv2 scene subset for VGGT training."
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output CO3D data directory.")
    parser.add_argument(
        "--annotation_dir",
        type=Path,
        required=True,
        help="Output directory for filtered VGGT .jgz annotations.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Archive/download cache directory; defaults to OUTPUT_DIR/.downloads.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="apple",
        help="Comma-separated eligible categories; default is one category to limit ZIP downloads.",
    )
    parser.add_argument("--max_images", type=int, default=20000, help="Maximum retained images over train+test.")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of image budget for test split.")
    parser.add_argument(
        "--max_frames_per_scene",
        type=int,
        default=100,
        help="Maximum retained images from any selected scene.",
    )
    parser.add_argument(
        "--min_frames_per_scene",
        type=int,
        default=24,
        help="Minimum retained frames per selected scene, matching VGGT CO3D default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_archives", action="store_true", help="Keep downloaded official ZIP chunks.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only download/read annotations and report the selected image budget.",
    )
    return parser.parse_args()


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using cached {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}\n  -> {destination}")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            written = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                written += len(chunk)
                if total and (written == len(chunk) or written == total or written % (256 * 1024 * 1024) < len(chunk)):
                    print(f"  {written / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} GiB")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def normalized_relative_path(path):
    relative = PurePosixPath(str(path).replace("\\", "/").lstrip("/"))
    if ".." in relative.parts:
        raise ValueError(f"Unsafe relative path in annotation: {path}")
    return relative.as_posix()


def load_or_download_annotations(categories, cache_dir):
    annotations = {}
    for category in categories:
        annotations[category] = {}
        for split in ("train", "test"):
            filename = f"{category}_{split}.jgz"
            path = cache_dir / "source_annotations" / filename
            download(VGGT_ANNO_URL.format(name=filename), path)
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                annotations[category][split] = json.load(stream)
    return annotations


def select_split(annotations, categories, split, budget, max_frames, min_frames, rng):
    candidates = []
    for category in categories:
        for sequence_name, frames in annotations[category][split].items():
            if len(frames) >= min_frames:
                candidates.append((category, sequence_name, frames))
    rng.shuffle(candidates)

    selected = defaultdict(dict)
    retained = 0
    for category, sequence_name, frames in candidates:
        remaining = budget - retained
        if remaining < min_frames:
            break
        count = min(len(frames), max_frames, remaining)
        if count < min_frames:
            continue
        if count < len(frames):
            frame_indices = sorted(rng.sample(range(len(frames)), count))
            selected_frames = [frames[index] for index in frame_indices]
        else:
            selected_frames = frames
        selected[category][sequence_name] = selected_frames
        retained += len(selected_frames)
        if retained >= budget:
            break
    return selected, retained


def required_assets(selected_by_split):
    required = defaultdict(set)
    for selected in selected_by_split.values():
        for category, sequences in selected.items():
            for frames in sequences.values():
                for frame in frames:
                    image_path = normalized_relative_path(frame["filepath"])
                    depth_path = image_path.replace("/images/", "/depths/") + ".geometric.png"
                    mask_path = image_path.replace("/images/", "/depth_masks/")
                    if mask_path.endswith(".jpg"):
                        mask_path = mask_path[:-4] + ".png"
                    required[category].update((image_path, depth_path, mask_path))
    return required


def member_to_required_path(member_name, wanted, wanted_by_basename):
    name = normalized_relative_path(member_name)
    if name in wanted:
        return name
    for candidate in wanted_by_basename.get(PurePosixPath(name).name, ()):
        if name.endswith("/" + candidate):
            return candidate
    return None


def extract_required_from_archive(archive, output_dir, missing):
    wanted_by_basename = defaultdict(list)
    for path in missing:
        wanted_by_basename[PurePosixPath(path).name].append(path)
    extracted = set()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue
            target_relative = member_to_required_path(member.filename, missing, wanted_by_basename)
            if target_relative is None:
                continue
            target = output_dir / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.add(target_relative)
    return extracted


def write_filtered_annotations(selected_by_split, categories, annotation_dir):
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for category in categories:
        for split in ("train", "test"):
            path = annotation_dir / f"{category}_{split}.jgz"
            content = selected_by_split[split].get(category, {})
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(content, stream)


def main():
    args = parse_args()
    if args.max_images < args.min_frames_per_scene * 2:
        raise ValueError("max_images must fit at least one train and one test scene.")
    if not 0 < args.val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    requested_categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    invalid = sorted(set(requested_categories) - set(VGGT_SEEN_CATEGORIES))
    if invalid:
        raise ValueError(f"Categories are not used by VGGT Co3dDataset: {invalid}")

    cache_dir = args.cache_dir or (args.output_dir / ".downloads")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_or_download_annotations(requested_categories, cache_dir)
    test_budget = int(round(args.max_images * args.val_fraction))
    train_budget = args.max_images - test_budget
    if test_budget < args.min_frames_per_scene:
        raise ValueError("Validation budget is smaller than min_frames_per_scene.")

    rng = random.Random(args.seed)
    selected_train, train_count = select_split(
        annotations, requested_categories, "train", train_budget,
        args.max_frames_per_scene, args.min_frames_per_scene, rng,
    )
    selected_test, test_count = select_split(
        annotations, requested_categories, "test", test_budget,
        args.max_frames_per_scene, args.min_frames_per_scene, rng,
    )
    if train_count == 0 or test_count == 0:
        raise RuntimeError("Could not select both train and test sequences with the requested settings.")

    selected_by_split = {"train": selected_train, "test": selected_test}
    required = required_assets(selected_by_split)
    print(f"Selected {train_count} train images and {test_count} test images ({train_count + test_count} total).")
    for category in requested_categories:
        train_scenes = len(selected_train.get(category, {}))
        test_scenes = len(selected_test.get(category, {}))
        if train_scenes or test_scenes:
            print(f"  {category}: {train_scenes} train scenes, {test_scenes} test scenes, {len(required[category]) // 3} images")
    if args.dry_run:
        return

    links_path = cache_dir / "links.json"
    download(CO3D_LINKS_URL, links_path)
    links = json.loads(links_path.read_text(encoding="utf-8"))["full"]

    for category, paths in required.items():
        missing = {path for path in paths if not (args.output_dir / path).exists()}
        if not missing:
            continue
        archive_urls = links.get(category, [])
        if not archive_urls:
            raise RuntimeError(f"No official archive URLs found for category {category}.")
        print(f"Extracting selected data for {category}; {len(missing)} files required.")
        for url in archive_urls:
            if not missing:
                break
            archive = cache_dir / Path(url).name
            download(url, archive)
            extracted = extract_required_from_archive(archive, args.output_dir, missing)
            missing.difference_update(extracted)
            print(f"  {archive.name}: extracted {len(extracted)} selected files, {len(missing)} remain")
            if not args.keep_archives:
                archive.unlink(missing_ok=True)
        if missing:
            examples = sorted(missing)[:5]
            raise RuntimeError(
                f"Official archives for {category} did not contain {len(missing)} required files. "
                f"Examples: {examples}. Check that VGGT annotations match downloaded CO3D version."
            )

    write_filtered_annotations(selected_by_split, requested_categories, args.annotation_dir)
    print(f"CO3D subset written to: {args.output_dir}")
    print(f"Filtered VGGT annotations written to: {args.annotation_dir}")
    print("Use these paths as CO3D_DIR and CO3D_ANNOTATION_DIR in the training config.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
