import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from sam_med2d_finetune.tools import run_yolo_retrain_pipeline as pipeline
from sam_med2d_finetune.tools.train_yolo import sha256_file


def _args(root):
    root = Path(root)
    data_yaml = root / "dataset" / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    model = root / "yolo11m.pt"
    model.write_bytes(b"base")
    return types.SimpleNamespace(
        data=str(data_yaml),
        model=str(model),
        project=str(root / "runs"),
        pipeline_dir=str(root / "pipeline"),
        python="python",
        device="0",
        ultralytics_dir=str(root / ".ultralytics"),
        seed=11171,
        workers=2,
        amp="true",
        skip_amp_check="true",
        screen_epochs=15,
        formal_epochs=100,
        patience=20,
        save_period=10,
        poll_seconds=0.01,
        deadline_hours=1.0,
        resume=False,
        command="run",
        shutdown_on_exit=False,
        shutdown_command="/usr/bin/shutdown",
        shutdown_grace_seconds=0.0,
    )


def _summary(passed):
    result = {
        "iou": 0.60,
        "conf": 0.001,
        "fully_missed_case_count": 0 if passed else 1,
        "missed_positive_slice_count_coverage_0.50": 0 if passed else 10,
        "max_consecutive_missed_positive_slices": 2 if passed else 3,
        "slice_coverage_recall_0.50": 0.98 if passed else 0.90,
        "background_false_positive_rate": 0.1,
        "mean_predicted_gt_box_area_ratio": 2.0,
    }
    return {"results": [result], "recommended": result}


class YoloRetrainPipelineTest(unittest.TestCase):
    def test_train_command_contains_recall_oriented_hyperparameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _args(temp_dir)
            config = pipeline.SCREEN_CONFIGS[1]

            command = pipeline.base_train_args(args, config, "screen_fixture", 15, config.fraction)

        self.assertIn("--optimizer", command)
        self.assertIn("SGD", command)
        self.assertIn("--nbs", command)
        self.assertIn("64", command)
        self.assertIn("--lr0", command)
        self.assertIn("0.01", command)
        self.assertIn("--mosaic", command)
        self.assertIn("0.0", command)
        self.assertIn("--scale", command)
        self.assertIn("0.2", command)
        self.assertIn("--box", command)
        self.assertIn("7.5", command)
        self.assertIn("--hsv_h", command)
        self.assertIn("--hsv_s", command)
        self.assertIn("--hsv_v", command)
        self.assertIn("0.1", command)
        self.assertIn("--fraction", command)
        self.assertIn(str(1.0 / 3.0), command)

    def test_y3_gate_requires_all_hard_conditions(self):
        self.assertTrue(pipeline.y3_gate_passed(_summary(True)["recommended"]))
        self.assertFalse(pipeline.y3_gate_passed(_summary(False)["recommended"]))

    def test_ready_marker_requires_remote_complete_and_hash_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_dir = Path(temp_dir) / "pipeline"
            pipeline_dir.mkdir()
            report = Path(temp_dir) / "report.md"
            report.write_text("evidence\n", encoding="utf-8")
            digest = sha256_file(report)

            with self.assertRaisesRegex(pipeline.PipelineError, "REMOTE_PIPELINE_COMPLETE"):
                pipeline.write_ready_marker(pipeline_dir, [f"{report}={digest}"])

            pipeline.write_marker(pipeline_dir, pipeline.REMOTE_COMPLETE_MARKER, {"status": "succeeded"})
            marker = pipeline.write_ready_marker(pipeline_dir, [f"{report}={digest}"])

            self.assertTrue(marker.is_file())

    def test_early_passing_512_screen_skips_remaining_screens_and_640(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _args(temp_dir)
            stages = []

            def fake_run_subprocess(command, log_path, pipeline_dir, stage, run_dir, poll_seconds, deadline_at):
                del log_path, pipeline_dir, poll_seconds, deadline_at
                stages.append(stage)
                if "train" in stage:
                    weights = Path(run_dir) / "weights"
                    weights.mkdir(parents=True, exist_ok=True)
                    (weights / "best.pt").write_bytes(b"best")
                    (weights / "last.pt").write_bytes(b"last")
                    (Path(run_dir) / "manifest.json").write_text("{}", encoding="utf-8")
                    return
                checkpoint = "best" if stage.endswith("_best") else "last"
                out_dir = Path(command[command.index("--out_dir") + 1])
                summary_dir = out_dir / f"{checkpoint}_val"
                summary_dir.mkdir(parents=True, exist_ok=True)
                (summary_dir / "scan_summary.json").write_text(
                    json.dumps(_summary(True)),
                    encoding="utf-8",
                )

            with mock.patch.object(pipeline, "run_subprocess", side_effect=fake_run_subprocess), mock.patch.object(
                pipeline,
                "collect_gpu_state",
                return_value=None,
            ):
                self.assertEqual(pipeline.run_pipeline(args), 0)

            joined = "\n".join(stages)
            self.assertIn("screen_train_s1_img512_mosaic1_scale0p5_box7p5", joined)
            self.assertNotIn("screen_train_s2_img512_mosaic0_scale0p2_box7p5", joined)
            self.assertNotIn("img640", joined)
            self.assertIn("formal_train", joined)
            self.assertTrue((Path(args.pipeline_dir) / pipeline.REMOTE_COMPLETE_MARKER).is_file())
            self.assertFalse((Path(args.pipeline_dir) / pipeline.FAILED_MARKER).is_file())

    def test_training_failure_does_not_start_y3(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _args(temp_dir)
            stages = []

            def fake_run_subprocess(command, log_path, pipeline_dir, stage, run_dir, poll_seconds, deadline_at):
                del command, log_path, pipeline_dir, run_dir, poll_seconds, deadline_at
                stages.append(stage)
                raise pipeline.PipelineError("synthetic training failure")

            with mock.patch.object(pipeline, "run_subprocess", side_effect=fake_run_subprocess):
                with self.assertRaisesRegex(pipeline.PipelineError, "synthetic training failure"):
                    pipeline.run_pipeline(args)

            self.assertEqual(stages, ["screen_train_s1_img512_mosaic1_scale0p5_box7p5"])

    def test_shutdown_request_persists_marker_before_exec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _args(temp_dir)
            args.shutdown_on_exit = True

            with mock.patch.object(pipeline.os, "sync", create=True), mock.patch.object(
                pipeline.time,
                "sleep",
            ), mock.patch.object(pipeline.os, "execv", return_value=None) as execv:
                self.assertTrue(pipeline.maybe_request_shutdown(args, 0, "pipeline_succeeded"))

            marker = Path(args.pipeline_dir) / pipeline.SHUTDOWN_REQUESTED_MARKER
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["exit_code"], 0)
            self.assertEqual(payload["reason"], "pipeline_succeeded")
            execv.assert_called_once_with("/usr/bin/shutdown", ["/usr/bin/shutdown"])

    def test_shutdown_exec_failure_persists_failure_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _args(temp_dir)
            args.shutdown_on_exit = True

            with mock.patch.object(pipeline.os, "sync", create=True), mock.patch.object(
                pipeline.time,
                "sleep",
            ), mock.patch.object(
                pipeline.os,
                "execv",
                side_effect=FileNotFoundError("missing shutdown"),
            ):
                self.assertFalse(pipeline.maybe_request_shutdown(args, 1, "pipeline_failed"))

            self.assertTrue((Path(args.pipeline_dir) / pipeline.SHUTDOWN_REQUESTED_MARKER).is_file())
            failed = json.loads(
                (Path(args.pipeline_dir) / pipeline.SHUTDOWN_FAILED_MARKER).read_text(encoding="utf-8")
            )
            self.assertEqual(failed["exit_code"], 1)
            self.assertEqual(failed["error"]["type"], "FileNotFoundError")

    def test_main_requests_shutdown_on_success_and_stops(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _args(temp_dir)
            argv = [
                "run_yolo_retrain_pipeline.py",
                "run",
                "--data",
                args.data,
                "--model",
                args.model,
                "--pipeline_dir",
                args.pipeline_dir,
                "--shutdown_on_exit",
                "--shutdown_grace_seconds",
                "0",
            ]

            with mock.patch("sys.argv", argv), mock.patch.object(
                pipeline,
                "run_pipeline",
                return_value=0,
            ), mock.patch.object(pipeline.os, "sync", create=True), mock.patch.object(
                pipeline.time,
                "sleep",
            ), mock.patch.object(pipeline.os, "execv", return_value=None) as execv:
                self.assertEqual(pipeline.main(), 0)

            execv.assert_called_once_with("/usr/bin/shutdown", ["/usr/bin/shutdown"])
            self.assertTrue((Path(args.pipeline_dir) / pipeline.SHUTDOWN_REQUESTED_MARKER).is_file())

    def test_main_requests_shutdown_on_training_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _args(temp_dir)
            argv = [
                "run_yolo_retrain_pipeline.py",
                "run",
                "--data",
                args.data,
                "--model",
                args.model,
                "--pipeline_dir",
                args.pipeline_dir,
                "--shutdown_on_exit",
                "--shutdown_grace_seconds",
                "0",
            ]

            with mock.patch("sys.argv", argv), mock.patch.object(
                pipeline,
                "run_pipeline",
                side_effect=pipeline.PipelineError("synthetic failure"),
            ), mock.patch.object(pipeline.os, "sync", create=True), mock.patch.object(
                pipeline.time,
                "sleep",
            ), mock.patch.object(pipeline.os, "execv", return_value=None) as execv:
                self.assertEqual(pipeline.main(), 1)

            execv.assert_called_once_with("/usr/bin/shutdown", ["/usr/bin/shutdown"])
            self.assertTrue((Path(args.pipeline_dir) / pipeline.FAILED_MARKER).is_file())
            self.assertTrue((Path(args.pipeline_dir) / pipeline.SHUTDOWN_REQUESTED_MARKER).is_file())

    def test_main_requests_shutdown_on_formal_y3_gate_failure_without_failed_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _args(temp_dir)
            argv = [
                "run_yolo_retrain_pipeline.py",
                "run",
                "--data",
                args.data,
                "--model",
                args.model,
                "--pipeline_dir",
                args.pipeline_dir,
                "--shutdown_on_exit",
                "--shutdown_grace_seconds",
                "0",
            ]

            with mock.patch("sys.argv", argv), mock.patch.object(
                pipeline,
                "run_pipeline",
                side_effect=pipeline.FormalY3GateFailure("gate failed"),
            ), mock.patch.object(pipeline.os, "sync", create=True), mock.patch.object(
                pipeline.time,
                "sleep",
            ), mock.patch.object(pipeline.os, "execv", return_value=None) as execv:
                self.assertEqual(pipeline.main(), 2)

            execv.assert_called_once_with("/usr/bin/shutdown", ["/usr/bin/shutdown"])
            self.assertFalse((Path(args.pipeline_dir) / pipeline.FAILED_MARKER).is_file())
            self.assertTrue((Path(args.pipeline_dir) / pipeline.SHUTDOWN_REQUESTED_MARKER).is_file())

    def test_status_does_not_request_shutdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "run_yolo_retrain_pipeline.py",
                "status",
                "--pipeline_dir",
                str(Path(temp_dir) / "pipeline"),
            ]

            with mock.patch("sys.argv", argv), mock.patch.object(
                pipeline.os,
                "execv",
                return_value=None,
            ) as execv, mock.patch.object(pipeline, "collect_gpu_state", return_value=None):
                self.assertEqual(pipeline.main(), 0)

            execv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
