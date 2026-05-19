# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import os.path as osp
import struct

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2


# COLMAP model id -> (name, params). Parameter order follows COLMAP sensor/models.h.
COLMAP_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", ("f", "cx", "cy")),
    1: ("PINHOLE", ("fx", "fy", "cx", "cy")),
    2: ("SIMPLE_RADIAL", ("f", "cx", "cy", "k1")),
    3: ("RADIAL", ("f", "cx", "cy", "k1", "k2")),
    4: ("OPENCV", ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")),
    5: ("OPENCV_FISHEYE", ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")),
    6: ("FULL_OPENCV", ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")),
    7: ("FOV", ("fx", "fy", "cx", "cy", "omega")),
    8: ("SIMPLE_RADIAL_FISHEYE", ("f", "cx", "cy", "k1")),
    9: ("RADIAL_FISHEYE", ("f", "cx", "cy", "k1", "k2")),
    10: ("THIN_PRISM_FISHEYE", ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", ("fx", "fy", "cx", "cy", "k0", "k1", "k2", "k3", "k4", "k5", "p0", "p1", "sx0", "sy0", "sx1", "sy1")),
    12: ("SIMPLE_DIVISION", ("f", "cx", "cy", "k")),
    13: ("DIVISION", ("fx", "fy", "cx", "cy", "k")),
    14: ("SIMPLE_FISHEYE", ("f", "cx", "cy")),
    15: ("FISHEYE", ("fx", "fy", "cx", "cy")),
}
COLMAP_CAMERA_NAME_TO_ID = {name: model_id for model_id, (name, _) in COLMAP_CAMERA_MODELS.items()}


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def _read_next_bytes(fid, num_bytes, fmt):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file")
    return struct.unpack("<" + fmt, data)


def _read_null_terminated_string(fid):
    chars = []
    while True:
        c = fid.read(1)
        if c == b"\x00" or c == b"":
            break
        chars.append(c)
    return b"".join(chars).decode("utf-8")


def _read_colmap_cameras_text(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            params = np.array([float(x) for x in parts[4:]], dtype=np.float32)
            cameras[camera_id] = {
                "model": model,
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": params,
            }
    return cameras


def _read_colmap_images_text(path):
    images = []
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for idx in range(0, len(lines), 2):
        parts = lines[idx].split()
        images.append(
            {
                "image_id": int(parts[0]),
                "qvec": np.array([float(x) for x in parts[1:5]], dtype=np.float32),
                "tvec": np.array([float(x) for x in parts[5:8]], dtype=np.float32),
                "camera_id": int(parts[8]),
                "name": " ".join(parts[9:]),
            }
        )
    return images


def _read_colmap_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            model, param_names = COLMAP_CAMERA_MODELS[model_id]
            params = np.array(_read_next_bytes(fid, 8 * len(param_names), "d" * len(param_names)), dtype=np.float32)
            cameras[camera_id] = {
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def _read_colmap_images_binary(path):
    images = []
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = _read_next_bytes(fid, 4, "i")[0]
            qvec = np.array(_read_next_bytes(fid, 32, "dddd"), dtype=np.float32)
            tvec = np.array(_read_next_bytes(fid, 24, "ddd"), dtype=np.float32)
            camera_id = _read_next_bytes(fid, 4, "i")[0]
            name = _read_null_terminated_string(fid)
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, os.SEEK_CUR)
            images.append(
                {
                    "image_id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name,
                }
            )
    return images


def _camera_params_to_intrinsic_k1(camera):
    model = camera["model"]
    params = camera["params"]

    if model in (
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
        "SIMPLE_DIVISION",
        "SIMPLE_FISHEYE",
    ):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in (
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
        "RAD_TAN_THIN_PRISM_FISHEYE",
        "DIVISION",
        "FISHEYE",
    ):
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")

    param_names = COLMAP_CAMERA_MODELS[COLMAP_CAMERA_NAME_TO_ID[model]][1]
    if "k1" in param_names:
        k1 = params[param_names.index("k1")]
    elif "k0" in param_names:
        k1 = params[param_names.index("k0")]
    else:
        k1 = 0.0

    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return intrinsic, np.array([k1], dtype=np.float32)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def _build_image_index(search_root):
    image_index = {}
    if not osp.isdir(search_root):
        return image_index
    for dirpath, _, filenames in os.walk(search_root):
        for filename in filenames:
            if osp.splitext(filename)[1] not in IMAGE_EXTENSIONS:
                continue
            full_path = osp.join(dirpath, filename)
            rel_path = osp.relpath(full_path, search_root)
            image_index.setdefault(filename, full_path)
            image_index.setdefault(rel_path, full_path)
            image_index.setdefault(rel_path.replace("\\", "/"), full_path)
    return image_index


def _find_image_path(image_roots, image_name, image_index=None):
    candidates = [osp.join(root, image_name) for root in image_roots]
    candidates += [osp.join(root, osp.basename(image_name)) for root in image_roots]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate

    if image_index is not None:
        basename = osp.basename(image_name)
        normalized = image_name.replace("\\", "/")
        if normalized in image_index:
            return image_index[normalized]
        if basename in image_index:
            return image_index[basename]

    raise FileNotFoundError(f"Could not resolve image '{image_name}' in roots: {image_roots}")


def _nerf_c2w_to_opencv_w2c(transform_matrix):
    c2w = np.array(transform_matrix, dtype=np.float32)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    return w2c[:3].astype(np.float32)


def _load_colmap_scene(sparse_dir, image_roots, scene_name):
    cameras_txt = osp.join(sparse_dir, "cameras.txt")
    images_txt = osp.join(sparse_dir, "images.txt")
    cameras_bin = osp.join(sparse_dir, "cameras.bin")
    images_bin = osp.join(sparse_dir, "images.bin")

    if osp.exists(cameras_txt) and osp.exists(images_txt):
        cameras = _read_colmap_cameras_text(cameras_txt)
        images = _read_colmap_images_text(images_txt)
    elif osp.exists(cameras_bin) and osp.exists(images_bin):
        cameras = _read_colmap_cameras_binary(cameras_bin)
        images = _read_colmap_images_binary(images_bin)
    else:
        return None

    image_index = _build_image_index(scene_name)
    frames = []
    for image in images:
        camera = cameras[image["camera_id"]]
        intrinsic, distortion = _camera_params_to_intrinsic_k1(camera)
        R = qvec_to_rotmat(image["qvec"])
        extrinsic = np.concatenate([R, image["tvec"][:, None]], axis=1).astype(np.float32)
        try:
            image_path = _find_image_path(image_roots, image["name"], image_index=image_index)
        except FileNotFoundError as exc:
            logging.warning(str(exc))
            continue
        frames.append(
            {
                "image_path": image_path,
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "distortion": distortion,
                "camera_model": camera["model"],
            }
        )

    return {"name": scene_name, "frames": frames} if frames else None


def _load_transforms_scene(path):
    root_dir = osp.dirname(path)
    with open(path, "r") as f:
        meta = json.load(f)

    fx = float(meta.get("fl_x"))
    fy = float(meta.get("fl_y", fx))
    cx = float(meta.get("cx", meta["w"] / 2.0))
    cy = float(meta.get("cy", meta["h"] / 2.0))
    k1 = float(meta.get("k1", 0.0))

    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    distortion = np.array([k1], dtype=np.float32)

    frames = []
    for frame in meta["frames"]:
        image_path = frame["file_path"]
        if image_path.startswith("./"):
            image_path = image_path[2:]
        frames.append(
            {
                "image_path": osp.join(root_dir, image_path),
                "extrinsic": _nerf_c2w_to_opencv_w2c(frame["transform_matrix"]),
                "intrinsic": intrinsic.copy(),
                "distortion": distortion.copy(),
                "camera_model": "transforms_json",
            }
        )

    return {"name": root_dir, "frames": frames} if frames else None


def _discover_robustvggt_scenes(root_dir, ignored_dir_names=("sparse.tmp",)):
    scenes = []
    seen_sparse_dirs = set()

    for dirpath, _, filenames in os.walk(root_dir):
        if any(part in ignored_dir_names for part in osp.normpath(dirpath).split(os.sep)):
            continue

        filenames = set(filenames)
        if "transforms.json" in filenames:
            scene = _load_transforms_scene(osp.join(dirpath, "transforms.json"))
            if scene is not None:
                scenes.append(scene)

        has_text = {"cameras.txt", "images.txt"}.issubset(filenames)
        has_bin = {"cameras.bin", "images.bin"}.issubset(filenames)
        if not (has_text or has_bin) or dirpath in seen_sparse_dirs:
            continue

        seen_sparse_dirs.add(dirpath)
        scene_root = osp.dirname(osp.dirname(dirpath)) if osp.basename(osp.dirname(dirpath)) == "sparse" else osp.dirname(dirpath)
        image_roots = [
            osp.join(scene_root, "images"),
            osp.join(scene_root, "rgb"),
            scene_root,
            root_dir,
        ]
        scene = _load_colmap_scene(dirpath, image_roots, scene_root)
        if scene is not None:
            scenes.append(scene)

    return scenes


class RobustVGGTK1Dataset(BaseDataset):
    """
    Minimal camera-only loader for onground/robustvggt.

    It reads COLMAP text/bin scenes and NeRF-style transforms.json files, then
    maps every camera model to the current 10D target by keeping only k1.
    """

    def __init__(
        self,
        common_conf,
        root_dir: str,
        split: str = "train",
        len_train: int = 100000,
        len_test: int = 10000,
        val_fraction: float = 0.1,
        split_seed: int = 42,
    ):
        super().__init__(common_conf=common_conf)
        self.root_dir = root_dir
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.len_train = len_train if split == "train" else len_test
        scenes = _discover_robustvggt_scenes(root_dir)
        if val_fraction > 0 and len(scenes) > 1:
            rng = np.random.default_rng(split_seed)
            order = rng.permutation(len(scenes))
            val_count = max(1, int(round(len(scenes) * val_fraction)))
            val_indices = set(order[:val_count].tolist())
            if split == "train":
                scenes = [scene for idx, scene in enumerate(scenes) if idx not in val_indices]
            elif split in ("val", "test"):
                scenes = [scene for idx, scene in enumerate(scenes) if idx in val_indices]
            else:
                raise ValueError(f"Unsupported split: {split}")

        self.scenes = scenes
        if not self.scenes:
            raise ValueError(f"No robustvggt scenes found for split={split} under {root_dir}")
        logging.info(f"Loaded {len(self.scenes)} robustvggt scenes from {root_dir} for split={split}")

    def get_data(self, seq_index=None, img_per_seq=None, seq_name=None, ids=None, aspect_ratio=1.0):
        if self.inside_random and self.training:
            scene = self.scenes[np.random.randint(0, len(self.scenes))]
        else:
            scene = self.scenes[(seq_index or 0) % len(self.scenes)]

        frames = scene["frames"]
        if ids is None or len(ids) == 0:
            replace = self.allow_duplicate_img or img_per_seq > len(frames)
            ids = np.random.choice(len(frames), img_per_seq, replace=replace)
        else:
            ids = np.asarray(ids) % len(frames)

        target_image_shape = self.get_target_shape(aspect_ratio)
        target_h, target_w = int(target_image_shape[0]), int(target_image_shape[1])

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        distortions = []
        original_sizes = []

        for image_idx in ids:
            frame = frames[int(image_idx)]
            image = read_image_cv2(frame["image_path"])
            original_h, original_w = image.shape[:2]

            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            intrinsic = frame["intrinsic"].copy()
            intrinsic[0, :] *= target_w / original_w
            intrinsic[1, :] *= target_h / original_h

            dummy_depth = np.zeros((target_h, target_w), dtype=np.float32)
            dummy_points = np.zeros((target_h, target_w, 3), dtype=np.float32)
            valid_mask = np.ones((target_h, target_w), dtype=bool)

            images.append(image)
            depths.append(dummy_depth)
            extrinsics.append(frame["extrinsic"])
            intrinsics.append(intrinsic)
            distortions.append(frame["distortion"])
            cam_points.append(dummy_points)
            world_points.append(dummy_points)
            point_masks.append(valid_mask)
            original_sizes.append(np.array([original_h, original_w]))

        return {
            "seq_name": scene["name"],
            "ids": np.asarray(ids),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "distortions": distortions,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }


class SimpleRadialColmapDataset(RobustVGGTK1Dataset):
    """Backward-compatible name for configs that point at a single COLMAP scene."""

    pass
