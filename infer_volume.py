import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm

from brats_case import BraTSCase
from model_factory import load_multitask_model
from postprocess_3d import postprocess_brats_masks


CLASS_NAMES = ("ET", "TC", "WT")
DEFAULT_YOLO_CHECKPOINT = "workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt"
DEFAULT_YOLO_IMGSZ = 320


class FullImageBoxPromptProvider:
    def __init__(self, image_size):
        self.image_size = image_size

    def should_skip_slice(self, slice_index, brats_case):
        del slice_index, brats_case
        return False

    def get_boxes(self, class_index, slice_index, brats_case):
        del class_index, slice_index, brats_case
        return torch.tensor(
            [[[0.0, 0.0, float(self.image_size - 1), float(self.image_size - 1)]]],
            dtype=torch.float32,
        )


class UpperBoundPromptProvider:
    def __init__(self, image_size):
        self.image_size = image_size

    def should_skip_slice(self, slice_index, brats_case):
        if not brats_case.has_segmentation():
            raise ValueError("prompt_mode=upper_bound requires '*_seg.nii.gz' to be present in the case directory.")
        return not brats_case.slice_has_any_gt(slice_index)

    def get_boxes(self, class_index, slice_index, brats_case):
        if not brats_case.has_segmentation():
            raise ValueError("prompt_mode=upper_bound requires '*_seg.nii.gz' to be present in the case directory.")

        class_name = CLASS_NAMES[class_index]
        gt_box = brats_case.get_gt_box(class_name, slice_index, image_size=self.image_size)
        if gt_box is None:
            return None
        return torch.tensor([[gt_box]], dtype=torch.float32)


def _configure_ultralytics_env(config_dir=".ultralytics"):
    config_dir = Path(config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLOv8_DIR", str(config_dir))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(config_dir))
    return config_dir


class YoloBoxPromptProvider:
    def __init__(self, image_size, yolo_checkpoint, yolo_conf, device):
        self.image_size = image_size
        self.yolo_checkpoint = Path(yolo_checkpoint).resolve()
        self.yolo_conf = float(yolo_conf)
        self.device = self._normalize_device(device)
        self._slice_cache = {}

        if not self.yolo_checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.yolo_checkpoint}")

        _configure_ultralytics_env()
        from ultralytics import YOLO

        self.model = YOLO(str(self.yolo_checkpoint))

    @staticmethod
    def _normalize_device(device):
        device = str(device)
        if device == "cuda":
            return "0"
        if device.startswith("cuda:"):
            return device.split(":", 1)[1]
        return device

    def _predict_box(self, slice_index, brats_case):
        cache_key = (brats_case.case_id, int(slice_index))
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]

        pseudo_rgb = brats_case.get_pseudo_rgb_slice(slice_index)
        result = self.model.predict(
            source=pseudo_rgb,
            conf=self.yolo_conf,
            iou=0.60,
            imgsz=DEFAULT_YOLO_IMGSZ,
            device=self.device,
            max_det=1,
            save=False,
            verbose=False,
        )[0]

        box = None
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            top_index = int(np.argmax(confs))
            top_conf = float(confs[top_index])
            if top_conf >= self.yolo_conf:
                height, width = brats_case.shape[:2]
                scale_x = float(self.image_size) / float(width)
                scale_y = float(self.image_size) / float(height)
                x1, y1, x2, y2 = xyxy[top_index].tolist()
                box = [
                    float(np.clip(x1 * scale_x, 0.0, self.image_size - 1.0)),
                    float(np.clip(y1 * scale_y, 0.0, self.image_size - 1.0)),
                    float(np.clip(x2 * scale_x, 0.0, self.image_size - 1.0)),
                    float(np.clip(y2 * scale_y, 0.0, self.image_size - 1.0)),
                ]

        self._slice_cache[cache_key] = box
        return box

    def should_skip_slice(self, slice_index, brats_case):
        return self._predict_box(slice_index, brats_case) is None

    def get_boxes(self, class_index, slice_index, brats_case):
        del class_index
        box = self._predict_box(slice_index, brats_case)
        if box is None:
            return None
        return torch.tensor([[box]], dtype=torch.float32)


PROMPT_PROVIDERS = {
    "full_image_box": FullImageBoxPromptProvider,
    "upper_bound": UpperBoundPromptProvider,
    "yolo_box": YoloBoxPromptProvider,
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Whole-case inference for BraTS volumes.")
    parser.add_argument("--case_dir", required=True, help="BraTS case directory with 4 modality NIfTI files.")
    parser.add_argument("--output_dir", required=True, help="Directory for NIfTI outputs and case_meta.json.")
    parser.add_argument("--sam_checkpoint", required=True, help="Base SAM-Med2D checkpoint path.")
    parser.add_argument("--finetuned_checkpoint", required=True, help="Adapter .pth or LoRA adapter directory.")
    parser.add_argument("--finetune_method", required=True, choices=["adapter", "lora"])
    parser.add_argument("--prompt_mode", default="full_image_box", choices=sorted(PROMPT_PROVIDERS))
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--input_channels", type=int, default=4)
    parser.add_argument("--encoder_adapter", type=str_to_bool, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_amp", type=str_to_bool, default=True)
    parser.add_argument("--yolo_checkpoint", default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--yolo_conf", type=float, default=0.05)
    parser.add_argument("--postprocess", type=str_to_bool, default=False)
    parser.add_argument("--closing_radius", type=int, default=1)
    parser.add_argument("--opening_radius", type=int, default=1)
    parser.add_argument("--wt_keep_largest", type=str_to_bool, default=True)
    parser.add_argument("--keep_topk_tc", type=int, default=2)
    parser.add_argument("--keep_topk_et", type=int, default=2)
    parser.add_argument("--z_smooth_iterations", type=int, default=1)
    return parser.parse_args()


def build_prompt_provider(prompt_mode, image_size, yolo_checkpoint=None, yolo_conf=0.05, device="cpu"):
    provider_class = PROMPT_PROVIDERS[prompt_mode]
    if prompt_mode == "yolo_box":
        return provider_class(
            image_size=image_size,
            yolo_checkpoint=yolo_checkpoint,
            yolo_conf=yolo_conf,
            device=device,
        )
    return provider_class(image_size=image_size)


def run_volume_inference(model, brats_case, prompt_provider, image_size, threshold, device, use_amp):
    height, width, depth = brats_case.shape
    class_volumes = {
        class_name: np.zeros((height, width, depth), dtype=np.uint8)
        for class_name in CLASS_NAMES
    }

    amp_enabled = use_amp and device.type == "cuda"

    with torch.no_grad():
        for slice_index in tqdm(range(depth), desc=f"Infer {brats_case.case_id}"):
            if prompt_provider.should_skip_slice(slice_index=slice_index, brats_case=brats_case):
                continue

            slice_tensor = brats_case.get_slice_tensor(slice_index, image_size)
            input_tensor = torch.from_numpy(slice_tensor).unsqueeze(0).to(device=device, dtype=torch.float32)

            with autocast(device_type=device.type, enabled=amp_enabled):
                image_embeddings = model.image_encoder(input_tensor)
                dense_pe = model.prompt_encoder.get_dense_pe()

                for class_index, class_name in enumerate(CLASS_NAMES):
                    boxes = prompt_provider.get_boxes(
                        class_index=class_index,
                        slice_index=slice_index,
                        brats_case=brats_case,
                    )
                    if boxes is None:
                        continue
                    boxes = boxes.to(device=device, dtype=torch.float32)
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=boxes,
                        masks=None,
                    )
                    low_res_masks, _ = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    upscaled_masks = F.interpolate(
                        low_res_masks,
                        size=(image_size, image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    probability_map = torch.sigmoid(upscaled_masks)[0, 0].detach().cpu().numpy()

                    original_probability = cv2.resize(
                        probability_map,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    class_volumes[class_name][:, :, slice_index] = (
                        original_probability >= threshold
                    ).astype(np.uint8)

    return class_volumes


def build_combined_label(class_volumes):
    combined = np.zeros_like(class_volumes["ET"], dtype=np.uint8)
    combined[class_volumes["WT"] > 0] = 2
    combined[class_volumes["TC"] > 0] = 1
    combined[class_volumes["ET"] > 0] = 4
    return combined


def _to_json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(output_path, payload):
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(_to_json_compatible(payload), indent=2),
        encoding="utf-8",
    )


def save_mask_outputs(brats_case, output_dir, class_volumes, combined_label, prefix=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename_prefix = f"{prefix}_" if prefix else ""
    brats_case.save_nifti(class_volumes["ET"].astype(np.uint8), output_dir / f"{filename_prefix}ET.nii.gz")
    brats_case.save_nifti(class_volumes["TC"].astype(np.uint8), output_dir / f"{filename_prefix}TC.nii.gz")
    brats_case.save_nifti(class_volumes["WT"].astype(np.uint8), output_dir / f"{filename_prefix}WT.nii.gz")
    brats_case.save_nifti(combined_label.astype(np.uint8), output_dir / f"{filename_prefix}combined_label.nii.gz")


def save_case_meta(brats_case, output_dir, meta):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    brats_case.write_case_meta(output_dir, meta)


def build_postprocess_config(
    enabled,
    closing_radius,
    opening_radius,
    wt_keep_largest,
    keep_topk_tc,
    keep_topk_et,
    z_smooth_iterations,
):
    return {
        "enabled": bool(enabled),
        "closing_radius": int(closing_radius),
        "opening_radius": int(opening_radius),
        "wt_keep_largest": bool(wt_keep_largest),
        "keep_topk_tc": int(keep_topk_tc),
        "keep_topk_et": int(keep_topk_et),
        "z_smooth_iterations": int(z_smooth_iterations),
    }


def build_case_meta(
    brats_case,
    output_dir,
    prompt_mode,
    finetune_method,
    sam_checkpoint,
    finetuned_checkpoint,
    image_size,
    threshold,
    postprocess_config,
    postprocess_report_path=None,
    yolo_config=None,
):
    return {
        "case_id": brats_case.case_id,
        "case_dir": str(brats_case.case_dir.resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "shape": list(brats_case.shape),
        "affine": brats_case.affine.tolist(),
        "voxel_spacing": list(brats_case.header.get_zooms()[:3]),
        "class_order": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "combined_label_map": {
            "1": "NCR/NET (TC minus ET)",
            "2": "ED (WT minus TC)",
            "4": "ET",
        },
        "prompt_mode": prompt_mode,
        "finetune_method": finetune_method,
        "sam_checkpoint": str(Path(sam_checkpoint).resolve()),
        "finetuned_checkpoint": str(Path(finetuned_checkpoint).resolve()),
        "image_size": image_size,
        "threshold": threshold,
        "normalization": {
            "mode": "per_volume_minmax_nonzero",
            "modalities": brats_case.normalization_stats,
        },
        "modality_paths": {
            key: str(path.resolve()) for key, path in brats_case.modality_paths.items()
        },
        "segmentation_path": str(brats_case.segmentation_path.resolve()) if brats_case.segmentation_path else None,
        "postprocess": {
            **postprocess_config,
            "report_path": str(postprocess_report_path.resolve()) if postprocess_report_path else None,
        },
        "yolo": yolo_config,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device)

    brats_case = BraTSCase.from_dir(args.case_dir)
    prompt_provider = build_prompt_provider(
        args.prompt_mode,
        args.image_size,
        yolo_checkpoint=args.yolo_checkpoint,
        yolo_conf=args.yolo_conf,
        device=args.device,
    )
    model = load_multitask_model(
        model_type=args.model_type,
        image_size=args.image_size,
        sam_checkpoint=args.sam_checkpoint,
        finetune_method=args.finetune_method,
        finetuned_checkpoint=args.finetuned_checkpoint,
        device=device,
        input_channels=args.input_channels,
        encoder_adapter=args.encoder_adapter,
    )

    class_volumes = run_volume_inference(
        model=model,
        brats_case=brats_case,
        prompt_provider=prompt_provider,
        image_size=args.image_size,
        threshold=args.threshold,
        device=device,
        use_amp=args.use_amp,
    )
    combined_label = build_combined_label(class_volumes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mask_outputs(brats_case, output_dir, class_volumes, combined_label)

    saved_files = [
        "ET.nii.gz",
        "TC.nii.gz",
        "WT.nii.gz",
        "combined_label.nii.gz",
    ]

    postprocess_config = build_postprocess_config(
        enabled=args.postprocess,
        closing_radius=args.closing_radius,
        opening_radius=args.opening_radius,
        wt_keep_largest=args.wt_keep_largest,
        keep_topk_tc=args.keep_topk_tc,
        keep_topk_et=args.keep_topk_et,
        z_smooth_iterations=args.z_smooth_iterations,
    )

    postprocess_report_path = None
    if args.postprocess:
        postprocessed_volumes, postprocess_report = postprocess_brats_masks(
            class_volumes=class_volumes,
            closing_radius=args.closing_radius,
            opening_radius=args.opening_radius,
            wt_keep_largest=args.wt_keep_largest,
            keep_topk_tc=args.keep_topk_tc,
            keep_topk_et=args.keep_topk_et,
            z_smooth_iterations=args.z_smooth_iterations,
        )
        post_combined_label = build_combined_label(postprocessed_volumes)
        save_mask_outputs(
            brats_case,
            output_dir,
            postprocessed_volumes,
            post_combined_label,
            prefix="post",
        )

        postprocess_report = {
            "case_id": brats_case.case_id,
            "output_dir": str(output_dir.resolve()),
            **postprocess_report,
        }
        postprocess_report_path = output_dir / "postprocess_report.json"
        save_json(postprocess_report_path, postprocess_report)
        saved_files.extend([
            "post_ET.nii.gz",
            "post_TC.nii.gz",
            "post_WT.nii.gz",
            "post_combined_label.nii.gz",
            "postprocess_report.json",
        ])

    meta = build_case_meta(
        brats_case=brats_case,
        output_dir=output_dir,
        prompt_mode=args.prompt_mode,
        finetune_method=args.finetune_method,
        sam_checkpoint=args.sam_checkpoint,
        finetuned_checkpoint=args.finetuned_checkpoint,
        image_size=args.image_size,
        threshold=args.threshold,
        postprocess_config=postprocess_config,
        postprocess_report_path=postprocess_report_path,
        yolo_config={
            "checkpoint": str(Path(args.yolo_checkpoint).resolve()) if args.prompt_mode == "yolo_box" else None,
            "conf": float(args.yolo_conf) if args.prompt_mode == "yolo_box" else None,
            "imgsz": DEFAULT_YOLO_IMGSZ if args.prompt_mode == "yolo_box" else None,
        },
    )
    save_case_meta(brats_case, output_dir, meta)

    print(json.dumps({
        "case_id": brats_case.case_id,
        "output_dir": str(output_dir.resolve()),
        "saved_files": saved_files + ["case_meta.json"],
    }, indent=2))


if __name__ == "__main__":
    main()
