from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from sam_med2d_finetune.models.factory import _unwrap_state_dict, build_multitask_base_model


class ModelFactoryTest(unittest.TestCase):
    def test_unwrap_state_dict_removes_model_and_ddp_wrappers(self):
        weight = torch.tensor([1.0])
        state_dict = _unwrap_state_dict({"model": {"module.weight": weight}})

        self.assertEqual(set(state_dict), {"weight"})
        self.assertIs(state_dict["weight"], weight)

    def test_build_passes_checkpoint_once_and_adapts_input_channels(self):
        model = nn.Module()
        model.image_encoder = nn.Module()
        model.image_encoder.patch_embed = nn.Module()
        model.image_encoder.patch_embed.proj = nn.Conv2d(3, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            model.image_encoder.patch_embed.proj.weight.copy_(
                torch.tensor([[[[1.0]], [[2.0]], [[3.0]]], [[[4.0]], [[5.0]], [[6.0]]]])
            )

        builder = Mock(return_value=model)
        with patch("sam_med2d_finetune.models.factory.sam_model_registry", {"vit_b": builder}):
            result = build_multitask_base_model(
                model_type="vit_b",
                image_size=256,
                sam_checkpoint="base.pth",
                input_channels=4,
                encoder_adapter=True,
            )

        builder.assert_called_once()
        build_args = builder.call_args.args[0]
        self.assertIsInstance(build_args, SimpleNamespace)
        self.assertEqual(build_args.sam_checkpoint, "base.pth")
        projection = result.image_encoder.patch_embed.proj
        self.assertEqual(projection.in_channels, 4)
        expected = torch.tensor([[[[2.0]]], [[[5.0]]]]).repeat(1, 4, 1, 1)
        self.assertTrue(torch.equal(projection.weight, expected))


if __name__ == "__main__":
    unittest.main()
