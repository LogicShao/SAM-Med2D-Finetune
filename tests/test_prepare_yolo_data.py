import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from sam_med2d_finetune.tools.evaluate_yolo_recall import load_ground_truth
from sam_med2d_finetune.tools.prepare_yolo_data import main as prepare_yolo_main


class DummyBraTSCase:
    def __init__(self, case_dir, positive_slices=(1, 2)):
        self.case_dir = Path(case_dir)
        self.case_id = self.case_dir.name
        self.shape = (8, 8, 4)
        wt_volume = np.zeros(self.shape, dtype=np.uint8)
        for slice_index in positive_slices:
            wt_volume[2:5, 3:6, slice_index] = 1
        self.class_gt_volumes = {"WT": wt_volume}

    def has_segmentation(self):
        return True

    def get_pseudo_rgb_slice(self, slice_index):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :, 0] = int(slice_index) * 10
        image[:, :, 1] = 32
        image[:, :, 2] = 64
        return image


def _make_case(split_dir, case_id, positive_slices=(1, 2)):
    case_dir = Path(split_dir) / case_id
    case_dir.mkdir(parents=True)
    for suffix in ("t1", "t1ce", "t2", "flair", "seg"):
        (case_dir / f"{case_id}_{suffix}.nii.gz").write_bytes(b"placeholder")
    return case_dir


def _write_split_manifest(root, train_ids, val_ids):
    manifest = {
        "seed": 11171,
        "splits": {
            "train": {"count": len(train_ids), "case_ids": train_ids},
            "val": {"count": len(val_ids), "case_ids": val_ids},
            "test": {"count": 1, "case_ids": ["BraTS2021_99999"]},
        },
    }
    (Path(root) / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _run_prepare(split_root, out_dir, extra_args=None, case_positive_slices=None):
    argv = [
        "prepare_yolo_data.py",
        "--split_root",
        str(split_root),
        "--out_dir",
        str(out_dir),
        "--seed",
        "11171",
    ]
    if extra_args:
        argv.extend(extra_args)
    case_positive_slices = case_positive_slices or {}

    def from_dir(case_dir):
        case_dir = Path(case_dir)
        return DummyBraTSCase(case_dir, case_positive_slices.get(case_dir.name, (1, 2)))

    with mock.patch("sys.argv", argv), mock.patch(
        "sam_med2d_finetune.tools.prepare_yolo_data.BraTSCase.from_dir",
        side_effect=from_dir,
    ):
        prepare_yolo_main()


class PrepareYoloDataTest(unittest.TestCase):
    def test_prepare_exports_train_sampling_and_complete_val(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_root = root / "paper_v1"
            train_dir = split_root / "train"
            val_dir = split_root / "val"
            train_ids = ["BraTS2021_00001", "BraTS2021_00002"]
            val_ids = ["BraTS2021_00003"]
            for case_id in train_ids:
                _make_case(train_dir, case_id)
            _make_case(val_dir, val_ids[0])
            _write_split_manifest(split_root, train_ids, val_ids)

            out_dir = root / "brats_yolo"
            _run_prepare(split_root, out_dir)

            train_images = sorted((out_dir / "images" / "train").glob("*.png"))
            val_images = sorted((out_dir / "images" / "val").glob("*.png"))
            self.assertEqual(len(train_images), 6)
            self.assertEqual(len(val_images), 4)

            train_labels = sorted((out_dir / "labels" / "train").glob("*.txt"))
            train_nonempty = [path for path in train_labels if path.read_text(encoding="utf-8").strip()]
            self.assertEqual(len(train_nonempty), 4)

            for label_path in train_nonempty:
                parts = label_path.read_text(encoding="utf-8").strip().split()
                self.assertEqual(parts[0], "0")
                for value in map(float, parts[1:]):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

            data_yaml = yaml.safe_load((out_dir / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data_yaml["names"][0], "Tumor")

            manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["seed"], 11171)
            self.assertEqual(manifest["modalities"], ["t1ce", "t2", "flair"])
            self.assertEqual(manifest["exports"]["val"]["exported_slices"], 4)
            self.assertEqual(manifest["exports"]["train"]["exported_negative_slices"], 2)

    def test_train_val_overlap_is_rejected_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_root = root / "paper_v1"
            case_id = "BraTS2021_00001"
            _make_case(split_root / "train", case_id)
            _make_case(split_root / "val", case_id)
            _write_split_manifest(split_root, [case_id], [case_id])

            out_dir = root / "brats_yolo"
            with self.assertRaises(ValueError):
                _run_prepare(split_root, out_dir)
            self.assertFalse(out_dir.exists())

    def test_cli_rejects_test_split_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            argv = [
                "prepare_yolo_data.py",
                "--split_root",
                str(root),
                "--out_dir",
                str(root / "out"),
                "--splits",
                "test",
            ]
            with mock.patch("sys.argv", argv), self.assertRaises(SystemExit):
                prepare_yolo_main()

    def test_yolo_recall_ground_truth_ignores_empty_label_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            label_dir = Path(temp_dir)
            (label_dir / "positive.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
            (label_dir / "negative.txt").write_text("", encoding="utf-8")

            ground_truth = load_ground_truth(label_dir)

            self.assertIn("positive", ground_truth)
            self.assertNotIn("negative", ground_truth)


if __name__ == "__main__":
    unittest.main()
