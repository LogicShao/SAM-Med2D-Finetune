import unittest

import numpy as np

from sam_med2d_finetune.inference.config import build_yolo_prompt_config, normalize_class_prompt_strategies
from sam_med2d_finetune.inference.io import build_combined_label


class InferenceSharedTest(unittest.TestCase):
    def test_class_prompt_strategy_uses_global_default_and_class_override(self):
        strategies = normalize_class_prompt_strategies(
            prompt_box_strategy="top1",
            prompt_box_strategy_et="top2_merge",
        )

        self.assertEqual(strategies, {"ET": "top2_merge", "TC": "top1", "WT": "top1"})

    def test_yolo_config_is_omitted_for_non_yolo_prompt_modes(self):
        self.assertIsNone(
            build_yolo_prompt_config(
                prompt_mode="full_image_box",
                yolo_checkpoint="unused.pt",
                yolo_conf=0.05,
                yolo_iou=0.6,
                yolo_max_det=2,
                yolo_topk=2,
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
            )
        )

    def test_combined_label_preserves_brats_label_precedence(self):
        et = np.zeros((2, 2, 2), dtype=np.uint8)
        tc = np.zeros_like(et)
        wt = np.zeros_like(et)
        et[0, 0, 0] = 1
        tc[0, 0, 0] = 1
        tc[0, 0, 1] = 1
        wt[0, 0, 0:2] = 1
        wt[0, 1, 0] = 1

        combined = build_combined_label({"ET": et, "TC": tc, "WT": wt})

        self.assertEqual(int(combined[0, 0, 0]), 4)
        self.assertEqual(int(combined[0, 0, 1]), 1)
        self.assertEqual(int(combined[0, 1, 0]), 2)


if __name__ == "__main__":
    unittest.main()
