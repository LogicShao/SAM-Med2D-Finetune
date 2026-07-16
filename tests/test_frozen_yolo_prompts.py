import builtins
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sam_med2d_finetune.inference.config import build_yolo_prompt_config
from sam_med2d_finetune.inference.volume import build_prompt_provider


class FrozenYoloPromptProviderTest(unittest.TestCase):
    CASE_ID = "BraTS2021_00001"

    @classmethod
    def _slice(cls, z_index, boxes=None):
        return {
            "stem": f"{cls.CASE_ID}_z{z_index:03d}",
            "case_id": cls.CASE_ID,
            "z_index": z_index,
            "boxes": [] if boxes is None else boxes,
        }

    @staticmethod
    def _box(xywh=None):
        return {
            "xywh": [0.5, 0.5, 0.5, 0.25] if xywh is None else xywh,
            "confidence": 0.9,
            "class_id": 0,
        }

    @classmethod
    def _payload(cls, slices=None):
        slices = [cls._slice(0, [cls._box()]), cls._slice(1)] if slices is None else slices
        return {
            "schema_version": 1,
            "model": "best.pt",
            "model_sha256": "a" * 64,
            "dataset_manifest": "dataset_manifest.json",
            "dataset_manifest_sha256": "b" * 64,
            "case_count": 1,
            "case_ids": [cls.CASE_ID],
            "slice_count": len(slices),
            "imgsz": 320,
            "max_det": 1,
            "iou": 0.6,
            "conf": 0.01,
            "slices": slices,
        }

    @staticmethod
    def _write_payload(temp_dir, payload):
        path = Path(temp_dir) / "predictions.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_replays_shared_top1_box_without_importing_ultralytics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_payload(temp_dir, self._payload())
            real_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "ultralytics" or name.startswith("ultralytics."):
                    raise AssertionError("Frozen prediction replay must not import Ultralytics.")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=guarded_import):
                provider = build_prompt_provider(
                    "frozen_yolo_box",
                    image_size=256,
                    yolo_predictions=path,
                )

            case = SimpleNamespace(case_id=self.CASE_ID, shape=(240, 240, 2))
            class_boxes = [
                provider.get_boxes(class_index, slice_index=0, brats_case=case)
                for class_index in range(3)
            ]
            for boxes in class_boxes:
                np.testing.assert_allclose(
                    boxes.numpy(),
                    np.asarray([[[64.0, 96.0, 192.0, 160.0]]], dtype=np.float32),
                )
            self.assertIsNone(provider.get_boxes(0, slice_index=1, brats_case=case))

            report = provider.build_case_prompt_report(case)
            self.assertEqual(report["summary"]["slices_with_prompt"], 1)
            self.assertEqual(report["summary"]["slices_skipped"], 1)
            self.assertEqual(len(report["prediction_sha256"]), 64)

    def test_config_records_frozen_prediction_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_payload(temp_dir, self._payload())
            config = build_yolo_prompt_config(
                prompt_mode="frozen_yolo_box",
                yolo_checkpoint="unused.pt",
                yolo_conf=0.05,
                yolo_iou=0.6,
                yolo_max_det=1,
                yolo_topk=1,
                prompt_box_strategy="top1",
                prompt_box_strategy_et=None,
                prompt_box_strategy_tc=None,
                prompt_box_strategy_wt=None,
                top2_score_ratio=0.5,
                top2_area_ratio_min=0.1,
                top2_area_ratio_max=2.0,
                top2_iou_max=0.9,
                z_prompt_mode="none",
                z_smooth_window=1,
                z_fill_gap_max=1,
                z_center_shift_max=64.0,
                z_area_ratio_min=0.25,
                z_area_ratio_max=4.0,
                wt_continuity_enabled=False,
                wt_continuity_score_thresh=0.15,
                wt_continuity_center_shift_max=48.0,
                wt_continuity_area_ratio_min=0.5,
                wt_continuity_area_ratio_max=2.0,
                wt_continuity_mask_dilate_iters=1,
                wt_continuity_mask_blur_kernel=3,
                class_prompt_variant="baseline",
                et_prompt_variant="default",
                yolo_predictions=path,
            )

            self.assertEqual(config["mode"], "frozen_replay")
            self.assertEqual(config["predictions"], str(path.resolve()))
            self.assertEqual(set(config["box_strategy_by_class"].values()), {"top1"})

    def test_rejects_duplicate_and_missing_slice_entries(self):
        duplicate = self._slice(0, [self._box()])
        missing_middle = [self._slice(0, [self._box()]), self._slice(2)]
        payloads = {
            "duplicate": self._payload([duplicate, duplicate]),
            "missing_middle": self._payload(missing_middle),
        }

        for name, payload in payloads.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                path = self._write_payload(temp_dir, payload)
                with self.assertRaises(ValueError):
                    build_prompt_provider(
                        "frozen_yolo_box",
                        image_size=256,
                        yolo_predictions=path,
                    )

    def test_rejects_invalid_normalized_box(self):
        invalid = self._slice(0, [self._box([0.9, 0.5, 0.4, 0.25])])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_payload(temp_dir, self._payload([invalid]))
            with self.assertRaisesRegex(ValueError, "normalized image width"):
                build_prompt_provider(
                    "frozen_yolo_box",
                    image_size=256,
                    yolo_predictions=path,
                )

    def test_rejects_missing_case_and_case_depth_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_payload(temp_dir, self._payload())
            provider = build_prompt_provider(
                "frozen_yolo_box",
                image_size=256,
                yolo_predictions=path,
            )

            missing_case = SimpleNamespace(case_id="BraTS2021_99999", shape=(240, 240, 2))
            with self.assertRaisesRegex(KeyError, "do not contain case_id"):
                provider.get_boxes(0, slice_index=0, brats_case=missing_case)

            wrong_depth = SimpleNamespace(case_id=self.CASE_ID, shape=(240, 240, 3))
            with self.assertRaisesRegex(ValueError, "do not match case depth"):
                provider.get_boxes(0, slice_index=0, brats_case=wrong_depth)


if __name__ == "__main__":
    unittest.main()
