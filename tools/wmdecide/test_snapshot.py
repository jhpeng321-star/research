import base64
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib
import json
from types import SimpleNamespace

from .snapshot import ROOT, registered_root, snapshot, unpack


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        parent = Path("E:/Temp/opencode") if os.name == "nt" else Path("/tmp")
        self.assertTrue(parent.is_dir())
        self.scratch = Path(tempfile.mkdtemp(prefix="wmdecide-snapshot-synthetic-", dir=parent))
        raw = b'{"synthetic_plumbing_only": true}\n'
        entry = dict(sha256=hashlib.sha256(raw).hexdigest(), data=base64.b64encode(raw).decode())
        self.payload = dict(host="gpuR", root=ROOT, files={
            "wm2_shard0/IDENTITY.json": entry, "wm2_shard0/w969420_k0.json": entry})

    def test_transfer_preserves_bytes_and_refuses_existing_destination(self):
        receipt = unpack(self.payload, self.scratch/"snapshot")
        self.assertEqual((self.scratch/"snapshot/wm2_shard0/w969420_k0.json").read_bytes(),
                         b'{"synthetic_plumbing_only": true}\n')
        self.assertEqual(len(receipt["files"]), 2)
        self.assertEqual(receipt["collection"], "read_only_no_inference_or_remote_writes")
        with self.assertRaises(ValueError):
            unpack(self.payload, self.scratch/"snapshot")

    def test_paths_and_corruption_rejected_before_any_output(self):
        for index, name in enumerate(("../w969420_k0.json", "/wm2_shard0/w969420_k0.json",
                "wm2_shard0/code.py", "wm2_shard0/w969440_k0.json", "shard0/a/b.json",
                "shard0/IDENTITY.json", "wm2_shard0//IDENTITY.json",
                "wm2_shard0\\IDENTITY.json")):
            payload = copy.deepcopy(self.payload)
            payload["files"][name] = next(iter(payload["files"].values()))
            out = self.scratch/f"bad{index}"
            with self.assertRaises(ValueError):
                unpack(payload, out)
            self.assertFalse(out.exists())
        bad = copy.deepcopy(self.payload)
        bad["files"]["wm2_shard0/IDENTITY.json"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            unpack(bad, self.scratch/"corrupt")
        self.assertFalse((self.scratch/"corrupt").exists())
        bad = copy.deepcopy(self.payload)
        bad["files"] = {}
        with self.assertRaises(ValueError):
            unpack(bad, self.scratch/"empty")
        self.assertFalse((self.scratch/"empty").exists())

    def test_ssh_is_read_only_and_identity_bound(self):
        def response(payload):
            raw = json.dumps(payload).encode()
            return SimpleNamespace(returncode=0, stderr="", stdout=
                                   base64.b64encode(zlib.compress(raw)).decode())
        with patch("tools.wmdecide.snapshot.subprocess.run", return_value=
                   response(self.payload)) as run:
            snapshot("gpuR", ROOT, self.scratch/"ssh")
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                  "gpuR", "/root/miniconda3/bin/python -B -"])
        compile(kwargs["input"], "synthetic_remote_inventory", "exec")
        self.assertIn("read_bytes()", kwargs["input"])
        self.assertNotIn("write_bytes", kwargs["input"])
        self.assertNotIn("subprocess", kwargs["input"])
        bad = copy.deepcopy(self.payload)
        bad["host"] = "gpuA"
        with patch("tools.wmdecide.snapshot.subprocess.run", return_value=response(bad)):
            with self.assertRaises(ValueError):
                snapshot("gpuR", ROOT, self.scratch/"wronghost")
        self.assertFalse((self.scratch/"wronghost").exists())

    def test_only_registered_output_root_and_host_pairs(self):
        self.assertEqual(registered_root("gpuR", ROOT), ROOT)
        self.assertEqual(registered_root("gpuA", ROOT+"/int/WM-PHYS-FULL-v3-gpu"),
                         ROOT+"/int/WM-PHYS-FULL-v3-gpu")
        for host, root in (("gpuA", ROOT), ("gpuR", ROOT+"/code"),
                           ("gpuA", ROOT+"/runa/WM-PHYS-A2/../code"),
                           ("gpuQ", ROOT), ("gpuR", ROOT+"/int/WM-PHYS-FULL-v3-gpu")):
            with self.assertRaises(ValueError):
                registered_root(host, root)


if __name__ == "__main__":
    unittest.main()
