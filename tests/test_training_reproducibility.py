import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sam_med2d_finetune.training.train_multitask import save_epoch_snapshot, seed_everything, seed_worker
from sam_med2d_finetune.utils.cli import str_to_bool


class NullLogger:
    def info(self, *args, **kwargs):
        pass


class TrainingReproducibilityTest(unittest.TestCase):
    def test_boolean_cli_parser_handles_false_explicitly(self):
        self.assertFalse(str_to_bool("false"))
        self.assertTrue(str_to_bool("true"))

    def test_global_seed_repeats_python_numpy_and_torch_sequences(self):
        seed_everything(123, deterministic=True)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        seed_everything(123, deterministic=True)
        second = (random.random(), np.random.rand(), torch.rand(1).item())

        self.assertEqual(first, second)

    def test_epoch_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(2, 1)
            args = SimpleNamespace(finetune_method="adapter", save_epochs=[1])
            logger = NullLogger()

            save_epoch_snapshot(model, args, temp_dir, 1, logger)
            snapshot_path = Path(temp_dir) / "epoch_001.pth"
            self.assertTrue(snapshot_path.is_file())
            with self.assertRaises(FileExistsError):
                save_epoch_snapshot(model, args, temp_dir, 1, logger)

    def test_worker_seed_reaches_augmentation_transform(self):
        transform = mock.Mock()
        worker_info = SimpleNamespace(dataset=SimpleNamespace(transform=transform))
        with mock.patch("torch.initial_seed", return_value=12345), mock.patch(
            "torch.utils.data.get_worker_info", return_value=worker_info
        ):
            seed_worker(0)

        transform.set_random_seed.assert_called_once_with(12345)


if __name__ == "__main__":
    unittest.main()
