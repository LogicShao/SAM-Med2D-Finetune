import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from brats_cache import (
    CACHE_IMAGES_DTYPE,
    CACHE_IMAGES_FILENAME,
    CACHE_NORMALIZATION,
    CACHE_SCHEMA_VERSION,
    CACHE_SEGMENTATION_DTYPE,
    CACHE_SEGMENTATION_FILENAME,
    normalize_nonzero_volume,
    validate_cache_case,
)
from brats_case import _normalize_volume
from multitask_dataset import BraTSDataset
from training_profiler import parse_cuda_device_index


class CachedBraTSDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.data_root = root / "data"
        self.cache_root = root / "cache"
        self.case_id = "BraTS2021_00001"
        (self.data_root / self.case_id).mkdir(parents=True)
        cache_case_dir = self.cache_root / self.case_id
        cache_case_dir.mkdir(parents=True)

        images = np.zeros((4, 5, 8, 8), dtype=np.float16)
        images[:, 1] = 0.5
        images[:, 3] = 1.0
        segmentation = np.zeros((5, 8, 8), dtype=np.uint8)
        segmentation[1, 2:5, 2:5] = 4
        segmentation[3, 1:6, 1:6] = 2
        np.save(str(cache_case_dir / CACHE_IMAGES_FILENAME), images)
        np.save(str(cache_case_dir / CACHE_SEGMENTATION_FILENAME), segmentation)
        (cache_case_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "case_id": self.case_id,
                    "images_file": CACHE_IMAGES_FILENAME,
                    "segmentation_file": CACHE_SEGMENTATION_FILENAME,
                    "image_shape": list(images.shape),
                    "segmentation_shape": list(segmentation.shape),
                    "images_dtype": str(CACHE_IMAGES_DTYPE),
                    "segmentation_dtype": str(CACHE_SEGMENTATION_DTYPE),
                    "normalization": CACHE_NORMALIZATION,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cache_indexes_all_positive_and_sampled_negative_slices(self):
        dataset = BraTSDataset(
            data_path=str(self.data_root),
            image_size=8,
            mode="val",
            cache_root=str(self.cache_root),
            cache_max_cases=1,
            negative_to_positive_ratio=0.5,
            sample_seed=7,
        )

        self.assertEqual(len(dataset), 3)
        labels = [dataset[index]["label"] for index in range(len(dataset))]
        self.assertEqual(sum(float(label.sum()) == 0.0 for label in labels), 1)
        self.assertTrue(all(tuple(sample["image"].shape) == (4, 8, 8) for sample in [dataset[0], dataset[1]]))

    def test_random_negative_boxes_are_deterministic(self):
        dataset = BraTSDataset(
            data_path=str(self.data_root),
            image_size=8,
            mode="val",
            cache_root=str(self.cache_root),
            negative_to_positive_ratio=1.0,
            negative_prompt_box="random",
            sample_seed=11,
        )

        negative_index = next(index for index in range(len(dataset)) if dataset[index]["label"].sum() == 0)
        first = dataset[negative_index]["boxes"]
        second = dataset[negative_index]["boxes"]
        self.assertTrue(np.array_equal(first.numpy(), second.numpy()))
        self.assertGreater(float(first.sum()), 0.0)

    def test_parse_cuda_device_index(self):
        self.assertEqual(parse_cuda_device_index("cuda"), 0)
        self.assertEqual(parse_cuda_device_index("cuda:1"), 1)
        self.assertIsNone(parse_cuda_device_index("cpu"))

    def test_cache_normalization_matches_inference_contract_with_float16_tolerance(self):
        raw_volume = np.array(
            [
                [[0.0, 10.0], [20.0, 0.0]],
                [[30.0, 40.0], [0.0, 50.0]],
            ],
            dtype=np.float32,
        )
        cache_normalized, _ = normalize_nonzero_volume(raw_volume)
        inference_normalized, _ = _normalize_volume(raw_volume)

        self.assertTrue(np.array_equal(cache_normalized, inference_normalized))
        self.assertTrue(
            np.allclose(cache_normalized.astype(np.float16).astype(np.float32), inference_normalized, atol=1e-3)
        )

    def test_cache_metadata_contract_is_checked(self):
        cache_case = validate_cache_case(self.cache_root, self.case_id)
        self.assertEqual(cache_case["images"].dtype, CACHE_IMAGES_DTYPE)
        self.assertEqual(cache_case["segmentation"].dtype, CACHE_SEGMENTATION_DTYPE)

        metadata_path = self.cache_root / self.case_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["schema_version"] = CACHE_SCHEMA_VERSION + 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_cache_case(self.cache_root, self.case_id)


if __name__ == "__main__":
    unittest.main()
