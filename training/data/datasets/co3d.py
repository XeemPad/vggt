# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np
import torch


from data.dataset_util import *
from data.base_dataset import BaseDataset
from data.augmentation import apply_random_simple_radial_augmentation


SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

COLMAP_CAMERA_PARAM_NAMES = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k1"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
    "FULL_OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
    "FOV": ("fx", "fy", "cx", "cy", "omega"),
    "SIMPLE_RADIAL_FISHEYE": ("f", "cx", "cy", "k1"),
    "RADIAL_FISHEYE": ("f", "cx", "cy", "k1", "k2"),
    "THIN_PRISM_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"),
    "RAD_TAN_THIN_PRISM_FISHEYE": ("fx", "fy", "cx", "cy", "k0", "k1", "k2", "k3", "k4", "k5", "p0", "p1", "sx0", "sy0", "sx1", "sy1"),
    "SIMPLE_DIVISION": ("f", "cx", "cy", "k"),
    "DIVISION": ("fx", "fy", "cx", "cy", "k"),
    "SIMPLE_FISHEYE": ("f", "cx", "cy"),
    "FISHEYE": ("fx", "fy", "cx", "cy"),
}


def _as_scalar(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def _k1_from_mapping(mapping):
    if not isinstance(mapping, dict):
        return None

    for key in ("k1", "radial_distortion", "distortion_k1"):
        if key in mapping:
            return _as_scalar(mapping[key])

    for key in ("distortion", "distortions", "distortion_params", "extra_params"):
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, dict):
            for distortion_key in ("k1", "k0", "k"):
                if distortion_key in value:
                    return _as_scalar(value[distortion_key])
        else:
            scalar = _as_scalar(value)
            if scalar is not None:
                return scalar

    model = None
    for key in ("camera_model", "camera_type", "model"):
        if key in mapping and mapping[key] is not None:
            model = mapping[key]
            break

    params = None
    for key in ("camera_params", "params", "colmap_params"):
        if key in mapping and mapping[key] is not None:
            params = mapping[key]
            break

    if model is not None and params is not None:
        model = str(model).upper()
        param_names = COLMAP_CAMERA_PARAM_NAMES.get(model)
        if param_names is not None:
            params = np.asarray(params, dtype=np.float32).reshape(-1)
            for param_name in ("k1", "k0", "k"):
                if param_name in param_names:
                    index = param_names.index(param_name)
                    if index < len(params):
                        return float(params[index])

    for key in ("camera", "viewpoint", "intrinsics", "colmap_camera"):
        value = mapping.get(key)
        if isinstance(value, dict):
            k1 = _k1_from_mapping(value)
            if k1 is not None:
                return k1

    return None


def _extract_simple_radial_k1(anno):
    k1 = _k1_from_mapping(anno)
    if k1 is None:
        return None
    return np.array([k1], dtype=np.float32)


class Co3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        CO3D_DIR: str = None,
        CO3D_ANNOTATION_DIR: str = None,
        categories: list = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the Co3dDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            CO3D_DIR (str): Directory path to CO3D data.
            CO3D_ANNOTATION_DIR (str): Directory path to CO3D annotations.
            categories (list): Optional subset of CO3D categories to load.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If CO3D_DIR or CO3D_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.radial_distortion_aug = common_conf.augs.get("radial_distortion", None)

        if CO3D_DIR is None or CO3D_ANNOTATION_DIR is None:
            raise ValueError("Both CO3D_DIR and CO3D_ANNOTATION_DIR must be specified.")

        category = sorted(categories if categories is not None else SEEN_CATEGORIES)
        invalid_categories = sorted(set(category) - set(SEEN_CATEGORIES))
        if invalid_categories:
            raise ValueError(f"Unsupported CO3D categories: {invalid_categories}")

        if self.debug:
            category = ["apple"]

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"CO3D_DIR is {CO3D_DIR}")

        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR

        total_frame_num = 0

        for c in category:
            for split_name in split_name_list:
                annotation_file = osp.join(
                    self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz"
                )

                try:
                    with gzip.open(annotation_file, "r") as fin:
                        annotation = json.loads(fin.read())
                except FileNotFoundError:
                    logging.error(f"Annotation file not found: {annotation_file}")
                    continue

                for seq_name, seq_data in annotation.items():
                    if len(seq_data) < min_num_images:
                        continue
                    if seq_name in self.invalid_sequence:
                        continue
                    total_frame_num += len(seq_data)
                    self.data_store[seq_name] = seq_data

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D Data dataset length: {len(self)}")
        if self.sequence_list_len == 0:
            expected_files = [osp.join(self.CO3D_ANNOTATION_DIR, f"{c}_{split_name_list[0]}.jgz") for c in category]
            raise ValueError(
                f"No valid CO3D sequences loaded for split={split}. "
                f"Expected annotation files under {self.CO3D_ANNOTATION_DIR}: {expected_files}. "
                f"Each sequence must contain at least {min_num_images} retained frames. "
                "If a subset was generated, do not train from a --dry_run result and use the same categories."
            )

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            seq_name = self.sequence_list[seq_index % self.sequence_list_len]

        metadata = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.allow_duplicate_img
            )

        annos = [metadata[i] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        distortions = []
        image_paths = []
        original_sizes = []

        raw_images = []
        raw_depths = []
        raw_extrinsics = []
        raw_intrinsics = []
        raw_distortions = []
        raw_filepaths = []
        raw_image_paths = []
        raw_original_sizes = []

        for anno in annos:
            filepath = anno["filepath"]

            image_path = osp.join(self.CO3D_DIR, filepath)
            image = read_image_cv2(image_path)

            if self.load_depth:
                depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
                depth_map = read_depth(depth_path, 1.0)

                mvs_mask_path = image_path.replace(
                    "/images", "/depth_masks"
                ).replace(".jpg", ".png")
                mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE) > 128
                depth_map[~mvs_mask] = 0

                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            extri_opencv = np.array(anno["extri"])
            intri_opencv = np.array(anno["intri"])
            distortion = _extract_simple_radial_k1(anno)

            raw_images.append(image)
            raw_depths.append(depth_map)
            raw_extrinsics.append(extri_opencv)
            raw_intrinsics.append(intri_opencv)
            raw_distortions.append(distortion)
            raw_filepaths.append(filepath)
            raw_image_paths.append(image_path)
            raw_original_sizes.append(original_size)

        radial_distortion_applied = False
        if self.radial_distortion_aug is not None and self.radial_distortion_aug.get("enabled", False):
            valid_distortions = all(distortion is not None for distortion in raw_distortions)
            if not valid_distortions and self.radial_distortion_aug.get("synthetic_from_zero", False):
                raw_distortions = [
                    np.zeros(1, dtype=np.float32) if distortion is None else distortion
                    for distortion in raw_distortions
                ]
                valid_distortions = True

            same_shape = len({image.shape for image in raw_images}) == 1
            if valid_distortions and same_shape:
                image_tensor = torch.from_numpy(np.stack(raw_images).astype(np.float32)).permute(0, 3, 1, 2)
                intrinsics_tensor = torch.from_numpy(np.stack(raw_intrinsics).astype(np.float32))
                distortions_tensor = torch.from_numpy(np.stack(raw_distortions).astype(np.float32))
                image_tensor, distortions_tensor = apply_random_simple_radial_augmentation(
                    image_tensor,
                    intrinsics_tensor,
                    distortions_tensor,
                    probability=self.radial_distortion_aug.get("p", 0.5),
                    delta_range=self.radial_distortion_aug.get("delta_range", (-0.05, 0.05)),
                    shared=self.radial_distortion_aug.get("shared", True),
                    clamp_range=self.radial_distortion_aug.get("clamp_range", (-0.3, 0.3)),
                    num_iters=self.radial_distortion_aug.get("num_iters", 8),
                    padding_mode=self.radial_distortion_aug.get("padding_mode", "border"),
                )
                raw_images = [
                    np.rint(image.permute(1, 2, 0).numpy()).clip(0, 255).astype(np.uint8)
                    for image in image_tensor
                ]
                raw_distortions = [
                    distortion.numpy().astype(np.float32)
                    for distortion in distortions_tensor
                ]
                radial_distortion_applied = True
            elif valid_distortions:
                logging.warning(
                    f"Skipping pre-process radial augmentation for {seq_name}: selected frames have different shapes."
                )

        for image, depth_map, extri_opencv, intri_opencv, distortion, filepath, image_path, original_size in zip(
            raw_images,
            raw_depths,
            raw_extrinsics,
            raw_intrinsics,
            raw_distortions,
            raw_filepaths,
            raw_image_paths,
            raw_original_sizes,
        ):

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            if distortion is not None:
                distortions.append(distortion)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        set_name = "co3d"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        if len(distortions) == len(extrinsics):
            batch["distortions"] = distortions
        if radial_distortion_applied:
            batch["radial_distortion_applied"] = True
        return batch
