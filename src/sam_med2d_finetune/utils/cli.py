"""Shared command-line parsing helpers."""

import argparse


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_torch_device(device):
    import torch

    device = str(device)
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)
