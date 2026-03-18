from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from peft import PeftModel

from segment_anything import sam_model_registry


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    if not isinstance(state_dict, dict):
        raise TypeError("Expected checkpoint to contain a state_dict-like object.")

    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _load_checkpoint(path):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return _unwrap_state_dict(state_dict)


def _replace_patch_embed(model, input_channels):
    if input_channels == model.image_encoder.patch_embed.proj.in_channels:
        return

    original_proj = model.image_encoder.patch_embed.proj
    new_proj = nn.Conv2d(
        in_channels=input_channels,
        out_channels=original_proj.out_channels,
        kernel_size=original_proj.kernel_size,
        stride=original_proj.stride,
        padding=original_proj.padding,
        bias=(original_proj.bias is not None),
    )

    with torch.no_grad():
        avg_weights = original_proj.weight.detach().mean(dim=1, keepdim=True)
        new_proj.weight.copy_(avg_weights.repeat(1, input_channels, 1, 1))
        if original_proj.bias is not None:
            new_proj.bias.copy_(original_proj.bias.detach())

    model.image_encoder.patch_embed.proj = new_proj


def build_multitask_base_model(
    model_type,
    image_size,
    sam_checkpoint,
    input_channels=4,
    encoder_adapter=True,
):
    build_args = SimpleNamespace(
        image_size=image_size,
        sam_checkpoint=sam_checkpoint,
        encoder_adapter=encoder_adapter,
    )
    model = sam_model_registry[model_type](build_args)

    if sam_checkpoint:
        state_dict = _load_checkpoint(sam_checkpoint)
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            pass

    _replace_patch_embed(model, input_channels)
    return model


def load_multitask_model(
    model_type,
    image_size,
    sam_checkpoint,
    finetune_method,
    finetuned_checkpoint,
    device,
    input_channels=4,
    encoder_adapter=True,
):
    model = build_multitask_base_model(
        model_type=model_type,
        image_size=image_size,
        sam_checkpoint=sam_checkpoint,
        input_channels=input_channels,
        encoder_adapter=encoder_adapter,
    )

    if finetune_method == "adapter":
        state_dict = _load_checkpoint(finetuned_checkpoint)
        model.load_state_dict(state_dict, strict=True)
    elif finetune_method == "lora":
        adapter_path = Path(finetuned_checkpoint)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")
        model.image_encoder = PeftModel.from_pretrained(
            model.image_encoder,
            adapter_path,
            is_trainable=False,
        )
    else:
        raise ValueError(f"Unsupported finetune method: {finetune_method}")

    model = model.to(device)
    model.eval()
    return model
