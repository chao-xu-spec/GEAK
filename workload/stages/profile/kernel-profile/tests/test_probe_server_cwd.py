"""The probe server runs in the run directory, so what it writes relative to its cwd stays there."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "shared"))

import probe_server  # noqa: E402


class ProbeServerCwdTest(unittest.TestCase):
    def test_server_is_started_in_the_run_directory(self):
        seen = {}

        class Exited:
            returncode = 1

            def poll(self):
                return 1

        def popen(cmd, **kwargs):
            seen.update(kwargs)
            return Exited()

        with tempfile.TemporaryDirectory() as run_dir:
            server = probe_server.Server({"port": 1, "compile_watch": False}, run_dir)
            with patch.object(probe_server, "command", return_value=(["server"], {})), \
                 patch.object(probe_server.subprocess, "Popen", side_effect=popen):
                result = server.start("eager", ready_timeout=5)
            self.assertFalse(result["ok"])
            self.assertEqual(Path(seen["cwd"]).resolve(), Path(run_dir).resolve())


if __name__ == "__main__":
    unittest.main()
