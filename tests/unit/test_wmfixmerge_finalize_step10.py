from tools.wmfixmerge.finalize_step10 import OUT, later_worker


def test_worker_binding_rejects_peer_output_and_checkpoint():
    argv = ["python", "-m", "tools.wmfixmerge.panel_eval_v2", "--out", str(OUT),
            "--candidate", str(OUT / "step_000020.pt"), "--name", "step20_full_s0"]
    assert later_worker(argv)
    for index, value in ((4, "/peer"), (6, str(OUT / "step_000010.pt")),
                         (8, "step10_full_s0"), (2, "peer.module")):
        changed = argv.copy()
        changed[index] = value
        assert not later_worker(changed)
    assert not later_worker(["python"])
