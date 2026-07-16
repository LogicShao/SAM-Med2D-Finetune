import contextlib
import io
import json
import signal
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

from sam_med2d_finetune.tools import train_yolo
from sam_med2d_finetune.tools.evaluate_yolo_recall import (
    choose_topk,
    evaluate_predictions,
    write_prediction_export,
)


class YoloRecallProtocolTest(unittest.TestCase):
    def test_coverage_case_misses_and_prediction_export(self):
        gt_box = (0.5, 0.5, 0.4, 0.4)
        image_stems = [
            "BraTS2021_00001_z000",
            "BraTS2021_00001_z001",
            "BraTS2021_00001_z002",
            "BraTS2021_00001_z003",
            "BraTS2021_00002_z000",
            "BraTS2021_00002_z001",
        ]
        gt_by_stem = {
            stem: [gt_box]
            for stem in image_stems
            if not stem.endswith("z003")
        }
        predictions = {
            "BraTS2021_00001_z000": [gt_box],
            "BraTS2021_00001_z001": [(0.5, 0.5, 0.2, 0.2)],
            "BraTS2021_00001_z002": [
                {"xywh": list(gt_box), "confidence": 0.9, "class_id": 0}
            ],
            "BraTS2021_00001_z003": [gt_box],
            "BraTS2021_00002_z000": [],
            "BraTS2021_00002_z001": [],
        }

        metrics = evaluate_predictions(predictions, gt_by_stem, image_stems)

        self.assertEqual(metrics["num_positive_slices"], 5)
        self.assertEqual(metrics["num_negative_slices"], 1)
        self.assertAlmostEqual(metrics["slice_coverage_recall_0.50"], 0.4)
        self.assertAlmostEqual(metrics["slice_coverage_recall_0.80"], 0.4)
        self.assertEqual(metrics["missed_positive_slice_count_coverage_0.50"], 3)
        self.assertEqual(metrics["fully_missed_case_count"], 1)
        self.assertEqual(metrics["fully_missed_case_ids"], ["BraTS2021_00002"])
        self.assertEqual(metrics["max_consecutive_missed_positive_slices"], 2)
        self.assertAlmostEqual(metrics["background_false_positive_rate"], 1.0)
        self.assertAlmostEqual(metrics["mean_predicted_gt_box_area_ratio"], 0.75)

        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path, prediction_sha256 = write_prediction_export(
                temp_dir,
                predictions,
                image_stems,
                {
                    "model": "fixture.pt",
                    "dataset_root": "fixture",
                    "split": "val",
                    "imgsz": 320,
                    "max_det": 1,
                    "iou": 0.6,
                    "conf": 0.01,
                },
            )
            payload = json.loads(prediction_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["slice_count"], len(image_stems))
            self.assertEqual(payload["slices"][0]["case_id"], "BraTS2021_00001")
            self.assertEqual(len(prediction_sha256), 64)

    def test_shortlist_prioritizes_case_and_slice_coverage(self):
        stronger_coverage = {
            "fully_missed_case_count": 0,
            "missed_positive_slice_count_coverage_0.50": 2,
            "max_consecutive_missed_positive_slices": 1,
            "slice_coverage_recall_0.50": 0.95,
            "background_false_positive_rate": 0.2,
            "mean_predicted_gt_box_area_ratio": 1.2,
            "iou": 0.6,
            "conf": 0.01,
        }
        fully_missed_case = {
            **stronger_coverage,
            "fully_missed_case_count": 1,
            "missed_positive_slice_count_coverage_0.50": 1,
            "slice_coverage_recall_0.50": 0.99,
        }

        shortlist = choose_topk([fully_missed_case, stronger_coverage], top_k=1)

        self.assertIs(shortlist[0], stronger_coverage)


def _fake_ultralytics_module(fail=False):
    module = types.ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, model):
            self.model = model

        def train(self, **kwargs):
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            if fail:
                raise RuntimeError("synthetic training failure")
            weights_dir = save_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            (weights_dir / "best.pt").write_bytes(b"best")
            (weights_dir / "last.pt").write_bytes(b"last")
            return types.SimpleNamespace(save_dir=save_dir)

    module.YOLO = FakeYOLO
    return module


class YoloRunManifestTest(unittest.TestCase):
    def test_non_git_code_state_is_unknown_and_hashes_entrypoint(self):
        with mock.patch.object(train_yolo, "_run_git_command", return_value=None):
            state = train_yolo.collect_git_state()

        self.assertIsNone(state["revision"])
        self.assertIsNone(state["dirty_worktree"])
        self.assertIsNone(state["dirty_paths"])
        self.assertEqual(
            state["entrypoint"]["sha256"],
            train_yolo.sha256_file(train_yolo.__file__),
        )

    def _make_dataset(self, root):
        dataset_dir = Path(root) / "dataset"
        dataset_dir.mkdir(exist_ok=True)
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump({"path": str(dataset_dir), "train": "images/train", "val": "images/val"}),
            encoding="utf-8",
        )
        (dataset_dir / "dataset_manifest.json").write_text(
            json.dumps({
                "seed": 11171,
                "splits": ["train", "val"],
                "exports": {
                    "train": {"case_count": 2, "exported_slices": 4, "cases": [{"case_id": "a"}]},
                    "val": {"case_count": 1, "exported_slices": 3, "cases": [{"case_id": "b"}]},
                },
            }),
            encoding="utf-8",
        )
        return data_yaml

    def _run_main(self, root, run_name, fail=False, resume=False):
        data_yaml = self._make_dataset(root)
        model_path = Path(root) / "yolo11m.pt"
        model_path.write_bytes(b"base")
        project_dir = Path(root) / "runs"
        argv = [
            "train_yolo.py",
            "--data",
            str(data_yaml),
            "--model",
            str(model_path),
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--nbs",
            "64",
            "--optimizer",
            "SGD",
            "--lr0",
            "0.01",
            "--momentum",
            "0.9",
            "--warmup_bias_lr",
            "0.0",
            "--project",
            str(project_dir),
            "--name",
            run_name,
            "--seed",
            "11171",
            "--ultralytics_dir",
            str(Path(root) / "ultralytics"),
            "--mosaic",
            "0.0",
            "--scale",
            "0.2",
            "--box",
            "10.0",
            "--hsv_h",
            "0.0",
            "--hsv_s",
            "0.0",
            "--hsv_v",
            "0.1",
        ]
        if resume:
            argv.extend(["--resume", "true"])
        with mock.patch("sys.argv", argv), mock.patch.dict(
            sys.modules,
            {"ultralytics": _fake_ultralytics_module(fail=fail)},
        ), mock.patch.object(
            train_yolo,
            "patch_ultralytics_dataset_threadpool",
        ), mock.patch.object(
            train_yolo,
            "collect_git_state",
            return_value={"revision": "fixture", "dirty_worktree": False, "dirty_paths": []},
        ), mock.patch.object(
            train_yolo,
            "collect_environment",
            return_value={"python": "fixture"},
        ), contextlib.redirect_stdout(io.StringIO()):
            train_yolo.main()
        return project_dir / run_name / "manifest.json"

    def test_success_manifest_contains_terminal_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = self._run_main(temp_dir, "y2_success_seed11171")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["exit_status"], 0)
            self.assertEqual(manifest["dataset"]["dataset_seed"], 11171)
            self.assertNotIn("cases", manifest["dataset"]["export_summary"]["train"])
            self.assertEqual(len(manifest["artifact_hashes"]["best.pt"]["sha256"]), 64)
            self.assertEqual(manifest["configuration"]["train_kwargs"]["fraction"], 1.0)
            self.assertTrue(manifest["configuration"]["train_kwargs"]["val"])
            self.assertFalse(manifest["configuration"]["skip_amp_check"])
            self.assertEqual(
                manifest["configuration"]["train_kwargs"]["optimizer"],
                "SGD",
            )
            self.assertEqual(manifest["configuration"]["train_kwargs"]["lr0"], 0.01)
            self.assertEqual(manifest["configuration"]["train_kwargs"]["momentum"], 0.9)
            self.assertEqual(
                manifest["configuration"]["train_kwargs"]["warmup_bias_lr"],
                0.0,
            )
            self.assertEqual(manifest["configuration"]["train_kwargs"]["mosaic"], 0.0)
            self.assertEqual(manifest["configuration"]["train_kwargs"]["scale"], 0.2)
            self.assertEqual(manifest["configuration"]["train_kwargs"]["box"], 10.0)
            self.assertEqual(manifest["configuration"]["train_kwargs"]["hsv_h"], 0.0)
            self.assertEqual(manifest["configuration"]["train_kwargs"]["hsv_s"], 0.0)
            self.assertEqual(manifest["configuration"]["train_kwargs"]["hsv_v"], 0.1)
            self.assertEqual(manifest["configuration"]["train_kwargs"]["nbs"], 64)

    def test_skip_amp_check_requires_amp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_yaml = self._make_dataset(temp_dir)
            argv = [
                "train_yolo.py",
                "--data",
                str(data_yaml),
                "--project",
                str(Path(temp_dir) / "runs"),
                "--name",
                "invalid_amp_config",
                "--amp",
                "false",
                "--skip_amp_check",
                "true",
            ]
            with mock.patch("sys.argv", argv), self.assertRaisesRegex(
                ValueError,
                "requires --amp true",
            ):
                train_yolo.main()

    def test_auto_optimizer_rejects_silently_ignored_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_yaml = self._make_dataset(temp_dir)
            argv = [
                "train_yolo.py",
                "--data",
                str(data_yaml),
                "--project",
                str(Path(temp_dir) / "runs"),
                "--name",
                "invalid_auto_optimizer_config",
                "--optimizer",
                "auto",
                "--lr0",
                "0.01",
            ]
            with mock.patch("sys.argv", argv), self.assertRaisesRegex(
                ValueError,
                "require --optimizer other than auto",
            ):
                train_yolo.main()

    def test_skip_amp_check_patch_returns_true_without_auxiliary_model(self):
        from ultralytics.engine import trainer as trainer_module

        with mock.patch.object(trainer_module, "check_amp", return_value=False):
            train_yolo.patch_ultralytics_amp_check()
            self.assertTrue(trainer_module.check_amp(object()))

    def test_failure_updates_precreated_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "synthetic training failure"):
                self._run_main(temp_dir, "y2_failure_seed11171", fail=True)
            manifest_path = Path(temp_dir) / "runs" / "y2_failure_seed11171" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["exit_status"], 1)
            self.assertEqual(manifest["error"]["type"], "RuntimeError")

            resumed_manifest_path = self._run_main(
                temp_dir,
                "y2_failure_seed11171",
                resume=True,
            )
            resumed_manifest = json.loads(resumed_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(resumed_manifest["status"], "succeeded")
            self.assertEqual(
                [item["status"] for item in resumed_manifest["status_history"]],
                ["running", "failed", "running", "succeeded"],
            )
            self.assertEqual(
                resumed_manifest["status_history"][-2]["event"],
                "retry_no_checkpoint",
            )

    def test_termination_signal_maps_to_interrupted_exit_status(self):
        error = train_yolo.TrainingTermination(signal.SIGTERM)

        self.assertEqual(error.signum, signal.SIGTERM)
        self.assertIn(str(signal.SIGTERM), str(error))


if __name__ == "__main__":
    unittest.main()
