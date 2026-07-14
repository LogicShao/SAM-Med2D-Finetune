import unittest

import numpy as np

from sam_med2d_finetune.brats.metrics import (
    compute_binary_metrics,
    compute_hierarchy_violations,
    compute_hd95_mm,
    evaluate_brats_case,
)


class BraTSMetricsTest(unittest.TestCase):
    def test_identical_masks_have_perfect_overlap_and_zero_hd95(self):
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1

        metrics = compute_binary_metrics(mask, mask, (1.0, 1.0, 1.0))

        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["hd95_mm"], 0.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)

    def test_hd95_uses_native_spacing(self):
        gt = np.zeros((5, 5, 5), dtype=np.uint8)
        pred = np.zeros_like(gt)
        gt[1:3, 1:3, 1:3] = 1
        pred[2:4, 1:3, 1:3] = 1

        hd95 = compute_hd95_mm(pred, gt, (2.0, 1.0, 1.0))

        self.assertEqual(hd95, 2.0)

    def test_empty_mask_policy_is_explicit(self):
        empty = np.zeros((4, 4, 4), dtype=np.uint8)
        nonempty = empty.copy()
        nonempty[1, 1, 1] = 1

        both_empty = compute_binary_metrics(empty, empty, (1.0, 1.0, 1.0))
        false_positive = compute_binary_metrics(nonempty, empty, (1.0, 1.0, 1.0))

        self.assertEqual(both_empty["dice"], 1.0)
        self.assertIsNone(both_empty["hd95_mm"])
        self.assertIsNone(both_empty["sensitivity"])
        self.assertEqual(false_positive["dice"], 0.0)
        self.assertIsNone(false_positive["sensitivity"])
        self.assertLess(false_positive["specificity"], 1.0)

    def test_hierarchy_violations_are_counted_once(self):
        et = np.zeros((3, 3, 3), dtype=np.uint8)
        tc = np.zeros_like(et)
        wt = np.zeros_like(et)
        et[0, 0, 0] = 1
        tc[1, 1, 1] = 1
        wt[2, 2, 2] = 1

        violations = compute_hierarchy_violations({"ET": et, "TC": tc, "WT": wt})

        self.assertEqual(violations["et_outside_tc_voxels"], 1)
        self.assertEqual(violations["tc_outside_wt_voxels"], 1)
        self.assertEqual(violations["any_violation_voxels"], 2)
        self.assertTrue(violations["has_violation"])

    def test_case_evaluation_rejects_mismatched_grid(self):
        masks = {name: np.zeros((3, 3, 3), dtype=np.uint8) for name in ("ET", "TC", "WT")}
        gt_masks = {name: np.zeros((4, 4, 4), dtype=np.uint8) for name in ("ET", "TC", "WT")}

        with self.assertRaises(ValueError):
            evaluate_brats_case(masks, gt_masks, (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
