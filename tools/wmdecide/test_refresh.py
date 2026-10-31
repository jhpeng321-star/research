import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from .rbeval_adapter import load_harness, sha256
from .refresh import (BEGIN, END, commit_publication, handoff_text, refresh_once,
                      complete_signature, require_clean_branch, run)
from .report import write_json


class RefreshTests(unittest.TestCase):
    def setUp(self):
        parent = Path("E:/Temp/opencode") if os.name == "nt" else Path("/tmp")
        self.scratch = Path(tempfile.mkdtemp(prefix="wmdecide-refresh-synthetic-", dir=parent))

    def test_handoff_retains_history_and_replaces_own_block(self):
        receipt = dict(completed_utc="synthetic_test_time", models={
            "wm2": dict(status="DIAGNOSTIC_ONLY", worlds=[969420])})
        old = "# Handoff\n\nOwner text and prior evidence.\n"
        first = handoff_text(old, "new_version", receipt)
        second = handoff_text(first, "next_version", receipt)
        self.assertIn("Owner text and prior evidence.", second)
        self.assertEqual(second.count(BEGIN), 1)
        self.assertEqual(second.count(END), 1)
        self.assertNotIn("new_version", second)
        self.assertIn("next_version", second)
        with self.assertRaises(ValueError):
            handoff_text(old+BEGIN, "bad", receipt)

    def test_claim_is_exclusive_and_past_deadline_never_collects(self):
        claim = self.scratch/"claim.json"
        write_json(claim, dict(synthetic_existing_owner=True))
        until = datetime.fromtimestamp(4102444800, timezone.utc)
        with patch("tools.wmdecide.refresh.require_clean_branch"), \
                patch("tools.wmdecide.refresh.snapshot") as collect:
            with self.assertRaises(FileExistsError):
                run(self.scratch, "unused", self.scratch/"job", claim, until)
            self.assertFalse((self.scratch/"job").exists())
            self.assertEqual(json.loads(claim.read_text()), dict(synthetic_existing_owner=True))
            with self.assertRaises(ValueError):
                run(self.scratch, "unused", self.scratch/"job", claim,
                    datetime.fromtimestamp(0, timezone.utc))
            collect.assert_not_called()

    def test_git_guards_never_stage_foreign_changes(self):
        def staged(repo, *args):
            if args == ("diff", "--cached", "--name-only"):
                return "user_file.py"
            return ""
        with patch("tools.wmdecide.refresh.git", side_effect=staged) as command:
            with self.assertRaises(ValueError):
                commit_publication(self.scratch, "docs/generated")
            self.assertFalse(any(c.args[1] in ("add", "commit") for c in command.call_args_list))
        for branch, status in (("another-branch", ""), ("c4/rb-decide", "?? user_file.py")):
            with patch("tools.wmdecide.refresh.git", side_effect=[branch, status]):
                with self.assertRaises(ValueError):
                    require_clean_branch(self.scratch)

    def test_loop_stops_at_deadline_and_preserves_own_claim(self):
        (self.scratch/"HANDOFF.md").write_text("# Synthetic handoff\nOwner history.\n")
        receipt = dict(completed_utc="synthetic_test_time", models={
            "wm2": dict(status="DIAGNOSTIC_ONLY", worlds=[969420])})

        def cycle(eval_root, scratch, publication, previous=None):
            scratch.mkdir()
            return receipt

        claim, job = self.scratch/"loop_claim.json", self.scratch/"loop_job"
        with patch("tools.wmdecide.refresh.require_clean_branch"), \
                patch("tools.wmdecide.refresh.refresh_once", side_effect=cycle) as refresh, \
                patch("tools.wmdecide.refresh.commit_publication", return_value={}), \
                patch("tools.wmdecide.refresh.time.time", side_effect=[0, 0, 1001, 1001]), \
                patch("psutil.cpu_percent", return_value=50):
            run(self.scratch, "unused", job, claim,
                datetime.fromtimestamp(1000, timezone.utc))
        self.assertEqual(refresh.call_count, 1)
        self.assertTrue((job/"COMPLETE.json").is_file())
        self.assertTrue(claim.is_file())
        self.assertIn("Owner history.", (self.scratch/"HANDOFF.md").read_text())
        identity = json.loads(claim.read_text())
        self.assertEqual(identity["pid"], os.getpid())
        self.assertFalse(identity["inference"])
        self.assertFalse(identity["process_signalling"])

    def test_unchanged_completed_worlds_cache_ignores_partial_progress(self):
        files = {f"shard0/w969420_k{k}.json": str(k) for k in range(8)}
        files["shard0/IDENTITY.json"] = "synthetic_identity"
        first = complete_signature(files)
        files["shard0/w969421_k0.json"] = "partial_only"
        self.assertEqual(complete_signature(files), first)
        files["shard0/w969420_k0.json"] = "changed_complete_task"
        self.assertNotEqual(complete_signature(files), first)

    def test_actual_validator_refresh_matches_worlds_and_retains_blocked_model(self):
        root = Path(os.environ.get("WMS_RBEVAL_ROOT",
                    str(Path(__file__).resolve().parents[2].parent/"c4-rb-eval")))
        core, _ = load_harness(root)
        real = core.load(core.HERE/"real_observables.json")["episodes"]
        tasks = []
        # Copied real observables are explicitly synthetic test forecasts, NOT
        # evidence of model skill. Two worlds exercise actual matched admission.
        for seed in (969420, 969421):
            episodes = []
            for arm in core.ARMS:
                row = next(r for r in real if r["seed"] == seed and r["arm"] == arm)
                episodes.append(dict(arm=arm, initial_sha256=row["initial_sha256"],
                    bins={reg: dict(speed=row["bins"][reg]["speed"][6:],
                                    post_counts=row["bins"][reg]["post_counts"]) for reg in core.REGIONS},
                    clocks={h: {key: c[key] for key in
                        ("team_delay_vs", "road_delay_vs", "source_wait_vs")}
                        for h, c in row["clocks"].items()}))
            tasks.extend(dict(seed=seed, future=k, episodes=copy.deepcopy(episodes)) for k in range(8))
        write_json(self.scratch/"manifest.json", dict(model="synthetic-refresh-plumbing",
                   model_sha256=sha256(core.HERE/"core.py"),
                   config_sha256=sha256(core.HERE/"references.json"),
                   sources={str(core.HERE/"core.py"): sha256(core.HERE/"core.py")}, tasks=tasks))
        subprocess.run([sys.executable, "-B", str(core.HERE/"import_external.py"), "--manifest",
                        str(self.scratch/"manifest.json"), "--out", str(self.scratch/"normalized")],
                       check=True, capture_output=True, text=True, timeout=120)

        def synthetic_snapshot(host, producer, out):
            out.mkdir()
            shard = out/"shard0"
            shard.mkdir()
            names = ["IDENTITY.json"]
            if producer == "synthetic_baseline":
                names += [f"w{seed}_k{k}.json" for seed in (969420, 969421) for k in range(8)]
            elif producer == "synthetic_candidate":
                names += [f"w969421_k{k}.json" for k in range(8)]
            else:
                names += ["w969420_k0.json"]
            for name in names:
                (shard/name).write_bytes((self.scratch/"normalized"/name).read_bytes())
            receipt = dict(files={"shard0/"+name: sha256(shard/name) for name in names},
                           synthetic_plumbing_only=True)
            write_json(out/"SNAPSHOT.json", receipt)
            return receipt

        producers = (("wm2", "unused", "synthetic_baseline"),
                     ("candidate", "unused", "synthetic_candidate"),
                     ("blocked", "unused", "synthetic_partial"))
        with patch("tools.wmdecide.refresh.PRODUCERS", producers), \
                patch("tools.wmdecide.refresh.snapshot", side_effect=synthetic_snapshot):
            receipt = refresh_once(root, self.scratch/"cycle", self.scratch/"publication")
        self.assertEqual(receipt["models"]["wm2"]["worlds"], [969420, 969421])
        self.assertEqual(receipt["models"]["candidate"]["worlds"], [969421])
        self.assertEqual(receipt["models"]["blocked"]["status"], "BLOCKED_NO_COMPLETE_K8_WORLD")
        paired = core.load(self.scratch/"publication/matched_candidate/REPORTS.json")
        self.assertEqual([r["scored_worlds"] for r in paired["reports"]], [[969421], [969421]])
        for name, expected in core.load(self.scratch/"publication/FILES.json").items():
            self.assertEqual(sha256(self.scratch/"publication"/name), expected)
        self.assertIn("blocked: BLOCKED_NO_COMPLETE_K8_WORLD",
                      (self.scratch/"publication/INDEX.md").read_text())
        with patch("tools.wmdecide.refresh.PRODUCERS", producers), \
                patch("tools.wmdecide.refresh.snapshot", side_effect=synthetic_snapshot), \
                patch("tools.wmdecide.refresh.credibility", side_effect=AssertionError(
                    "identical all-world report should be cached")):
            reused = refresh_once(root, self.scratch/"cycle2", self.scratch/"publication2",
                                  previous=receipt["models"])
        self.assertEqual(reused["models"]["wm2"]["report"], receipt["models"]["wm2"]["report"])
        self.assertIn("reused_report", reused["models"]["wm2"])


if __name__ == "__main__":
    unittest.main()
