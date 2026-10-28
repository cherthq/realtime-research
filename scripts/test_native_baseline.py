"""Command-contract tests: no GPU, checkpoint download or inference."""

from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.run_native_baseline import build_command, stop_child


class NativeCommandTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.row = {
            "reference_image": "images/ref01.png",
            "audio": "audio/s01_greeting.wav",
            "prompt": "A seated person; this stays a literal argument, not a shell command.",
        }

    def command(self, model):
        return build_command(
            model,
            self.root / "env/bin/python",
            self.root / "source",
            self.root / "weights",
            self.root,
            self.row,
            self.root / "run",
            42,
            10.1,
        )

    def test_liveavatar_uses_local_lora_and_bounds_audio(self):
        command = self.command("liveavatar")
        self.assertIn("torch.distributed.run", command)
        self.assertEqual(
            command[command.index("--lora_path_dmd") + 1],
            str(self.root / "weights/Live-Avatar/liveavatar.safetensors"),
        )
        self.assertEqual(command[command.index("--num_clip") + 1], "7")
        self.assertEqual(command[command.index("--base_seed") + 1], "42")
        self.assertEqual(command[command.index("--prompt") + 1], self.row["prompt"])
        self.assertNotIn("--enable_tts", command)
        self.assertNotIn("--use_prompt_extend", command)
        self.assertNotIn("--fp8", command)
        self.assertIn("--offload_kv_cache", command)

    def test_liveact_records_single_gpu_condition(self):
        command = self.command("liveact")
        self.assertIn("--fp8_kv_cache", command)
        self.assertIn("--t5_cpu", command)
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertEqual(command[command.index("--fps") + 1], "24")
        self.assertNotIn("--fp8_gemm", command)
        self.assertNotIn("--steam_audio", command)

    def test_unknown_model_and_escaping_input_are_rejected(self):
        with self.assertRaises(ValueError):
            self.command("unknown")
        self.row["audio"] = "../outside.wav"
        with self.assertRaises(ValueError):
            self.command("liveavatar")


class CleanupTests(unittest.TestCase):
    @patch("scripts.run_native_baseline.os.killpg", create=True)
    def test_exited_child_is_not_signalled(self, kill):
        child = Mock()
        child.poll.return_value = 0
        stop_child(child)
        kill.assert_not_called()

    @patch("scripts.run_native_baseline.os.killpg", create=True)
    @patch("scripts.run_native_baseline.signal.SIGKILL", 9, create=True)
    def test_timeout_escalates_entire_process_group(self, kill):
        child = Mock(pid=123)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("model", 15), -9]
        stop_child(child)
        self.assertEqual(kill.call_args_list[0].args, (123, signal.SIGTERM))
        # Windows does not expose SIGKILL; this runner itself targets Linux.
        self.assertEqual(kill.call_count, 2)


if __name__ == "__main__":
    unittest.main()
