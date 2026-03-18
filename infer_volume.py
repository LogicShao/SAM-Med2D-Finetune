import argparse
import json
import logging
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


class FullImageBoxPromptProvider:
    def __init__(self, image_size):
        self.image_size = image_size

    def get_boxes(self, class_index, slice_index):
        del class_index, slice_index
        return torch.tensor(
            [[[0.0, 0.0, float(self.image_size - 1), float(self.image_size - 1)]]],
            dtype=torch.float32,
        )


PROMPT_PROVIDERS = {
    "full_image_box": FullImageBoxPromptProvider,
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
    parser.add_argument("--postprocess", type=str_to_bool, default=False)
    parser.add_argument("--closing_radius", type=int, default=1)
    parser.add_argument("--opening_radius", type=int, default=1)
    parser.add_argument("--wt_keep_largest", type=str_to_bool, default=True)
    parser.add_argument("--keep_topk_tc", type=int, default=2)
    parser.add_argument("--keep_topk_et", type=int, default=2)
    parser.add_argument("--z_smooth_iterations", type=int, default=1)
    return parser.parse_args()


def build_prompt_provider(prompt_mode, image_size):
    return PROMPT_PROVIDERS[prompt_mode](image_size=image_size)


def run_volume_inference(model, brats_case, prompt_provider, image_size, threshold, device, use_amp):
    height, width, depth = brats_case.shape
    class_volumes = {
        class_name: np.zeros((height, width, depth), dtype=np.uint8)
        for class_name in CLASS_NAMES
    }

    amp_enabled = use_amp and device.type == "cuda"

    with torch.no_grad():
        for slice_index in tqdm(range(depth), desc=f"Infer {brats_case.case_id}"):
            slice_tensor = brats_case.get_slice_tensor(slice_index, image_size)
            input_tensor = torch.from_numpy(slice_tensor).unsqueeze(0).to(device=device, dtype=torch.float32)

            with autocast(device_type=device.type, enabled=amp_enabled):
                image_embeddings = model.image_encoder(input_tensor)
                dense_pe = model.prompt_encoder.get_dense_pe()

                for class_index, class_name in enumerate(CLASS_NAMES):
                    boxes = prompt_provider.get_boxes(class_index=class_index, slice_index=slice_index)
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
        "postprocess": {
            **postprocess_config,
            "report_path": str(postprocess_report_path.resolve()) if postprocess_report_path else None,
        },
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device)

    brats_case = BraTSCase.from_dir(args.case_dir)
    prompt_provider = build_prompt_provider(args.prompt_mode, args.image_size)
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
    )
    save_case_meta(brats_case, output_dir, meta)

    print(json.dumps({
        "case_id": brats_case.case_id,
        "output_dir": str(output_dir.resolve()),
        "saved_files": saved_files + ["case_meta.json"],
    }, indent=2))


if __name__ == "__main__":
    main()
