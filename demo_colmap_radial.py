# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import glob
import os
import random

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import trimesh

from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
from vggt.dependency.track_predict import predict_tracks
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

DEFAULT_BASE_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT COLMAP demo with fine-tuned SIMPLE_RADIAL camera head")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing scene images/")
    parser.add_argument("--camera_checkpoint", type=str, required=True, help="Camera-only fine-tuned checkpoint")
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Full pretrained VGGT checkpoint path or URL",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ba", action="store_true", default=False)
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--shared_camera", action="store_true", default=False)
    parser.add_argument("--camera_type", type=str, default="SIMPLE_RADIAL")
    parser.add_argument("--vis_thresh", type=float, default=0.2)
    parser.add_argument("--query_frame_num", type=int, default=8)
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--fine_tracking", action="store_true", default=True)
    parser.add_argument("--conf_thres_value", type=float, default=5.0)
    parser.add_argument("--vggt_resolution", type=int, default=518)
    parser.add_argument("--load_resolution", type=int, default=1024)
    return parser.parse_args()


def _load_checkpoint(path_or_url):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return torch.hub.load_state_dict_from_url(path_or_url, map_location="cpu")
    return torch.load(path_or_url, map_location="cpu")


def _checkpoint_model_state(checkpoint):
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def _expand_9d_camera_tensors_for_k1(model, state_dict):
    current_state = model.state_dict()
    adapted = dict(state_dict)
    for key, old_value in list(state_dict.items()):
        if key not in current_state:
            continue
        new_value = current_state[key]
        if old_value.shape == new_value.shape or "camera_head" not in key:
            continue

        expanded = torch.zeros_like(new_value)
        if old_value.ndim == 3 and old_value.shape[-1] + 1 == new_value.shape[-1]:
            expanded[..., : old_value.shape[-1]] = old_value
        elif old_value.ndim == 2 and old_value.shape[1] + 1 == new_value.shape[1]:
            expanded[:, : old_value.shape[1]] = old_value
        elif old_value.ndim == 2 and old_value.shape[0] + 1 == new_value.shape[0]:
            expanded[: old_value.shape[0], :] = old_value
        elif old_value.ndim == 1 and old_value.shape[0] + 1 == new_value.shape[0]:
            expanded[: old_value.shape[0]] = old_value
        else:
            continue

        adapted[key] = expanded
        print(f"Expanded {key}: {tuple(old_value.shape)} -> {tuple(new_value.shape)}")
    return adapted


def load_vggt_with_radial_camera_head(base_checkpoint, camera_checkpoint, device):
    model = VGGT(camera_pose_encoding_type="absT_quaR_FoV_k1")

    base_state = _checkpoint_model_state(_load_checkpoint(base_checkpoint))
    base_state = _expand_9d_camera_tensors_for_k1(model, base_state)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if missing:
        print(f"Base checkpoint missing keys: {missing}")
    if unexpected:
        print(f"Base checkpoint unexpected keys: {unexpected}")

    camera_state = _checkpoint_model_state(_load_checkpoint(camera_checkpoint))
    camera_state = {
        key.removeprefix("module."): value
        for key, value in camera_state.items()
        if key.startswith("camera_head.") or key.startswith("module.camera_head.")
    }
    if not camera_state:
        raise ValueError(f"No camera_head tensors found in {camera_checkpoint}")
    missing, unexpected = model.load_state_dict(camera_state, strict=False)
    print(f"Loaded {len(camera_state)} camera_head tensors from {camera_checkpoint}")
    if unexpected:
        print(f"Camera checkpoint unexpected keys: {unexpected}")

    model.eval()
    return model.to(device)


def run_vggt_radial(model, images, dtype, resolution=518):
    assert len(images.shape) == 4
    assert images.shape[1] == 3
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic, distortion = pose_encoding_to_extri_intri(
            pose_enc,
            images.shape[-2:],
            pose_encoding_type="absT_quaR_FoV_k1",
            return_distortion=True,
        )
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    return (
        extrinsic.squeeze(0).cpu().numpy(),
        intrinsic.squeeze(0).cpu().numpy(),
        distortion.squeeze(0).cpu().numpy(),
        depth_map.squeeze(0).cpu().numpy(),
        depth_conf.squeeze(0).cpu().numpy(),
    )


def demo_fn(args):
    print("Arguments:", vars(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    model = load_vggt_with_radial_camera_head(args.base_checkpoint, args.camera_checkpoint, device)
    print("Model loaded")

    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    images, original_coords = load_and_preprocess_images_square(image_path_list, args.load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    extrinsic, intrinsic, distortion, depth_map, depth_conf = run_vggt_radial(
        model, images, dtype, args.vggt_resolution
    )
    print(f"Predicted k1 range: min={distortion[:, 0].min():.6f}, max={distortion[:, 0].max():.6f}")

    points_3d_map = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = args.load_resolution / args.vggt_resolution
        shared_camera = args.shared_camera
        intrinsic[:, :2, :] *= scale

        with torch.cuda.amp.autocast(dtype=dtype):
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d_map,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )
            torch.cuda.empty_cache()

        track_mask = pred_vis_scores > args.vis_thresh
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            extra_params=distortion,
            points_rgb=points_rgb,
        )
        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)
        reconstruction_resolution = args.load_resolution
    else:
        max_points_for_colmap = 100000
        shared_camera = False
        camera_type = "SIMPLE_RADIAL"
        image_size = np.array([args.vggt_resolution, args.vggt_resolution])
        num_frames, height, width, _ = points_3d_map.shape

        points_rgb = F.interpolate(
            images, size=(args.vggt_resolution, args.vggt_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
        conf_mask = randomly_limit_trues(depth_conf >= args.conf_thres_value, max_points_for_colmap)

        points_3d = points_3d_map[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP SIMPLE_RADIAL format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
            extra_params=distortion,
        )
        reconstruction_resolution = args.vggt_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse_radial")
    print(f"Saving reconstruction to {sparse_reconstruction_dir}")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)
    np.save(os.path.join(sparse_reconstruction_dir, "predicted_k1.npy"), distortion[:, 0])
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(sparse_reconstruction_dir, "points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size

            camera_model_name = pycamera.model.name if hasattr(pycamera.model, "name") else str(pycamera.model)
            if "SIMPLE_RADIAL" in camera_model_name:
                pred_params[0] *= resize_ratio
                pred_params[1:3] = real_image_size / 2
            else:
                pred_params = pred_params * resize_ratio
                pred_params[-2:] = real_image_size / 2

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
