"""A resumed Adam update equals uninterrupted training, on CPU only."""
import pytest

torch = pytest.importorskip("torch")

from tools.wmfixmerge.data import digest
from tools.wmfixmerge.data_pilot import restore_completed


def test_exact_adam_resume_and_frozen_rejection(tmp_path):
    old, out, bad = (tmp_path / name for name in ("old", "new", "bad"))
    for path in (old, out, bad):
        path.mkdir()
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    model.bias.requires_grad_(False)
    frozen = {"bias": model.bias.detach().clone()}
    opt = torch.optim.Adam([model.weight], lr=3e-4)
    def update(m, o):
        o.zero_grad()
        m(torch.ones(1, 2)).square().sum().backward()
        o.step()
    for _ in range(4):
        update(model, opt)
    torch.save({"model": model.state_dict()}, old / "step_000004.pt")
    torch.save(opt.state_dict(), old / "optimizer_000004.pt")
    recovery = dict(previous_out=str(old), completed_steps=4,
                    resume_files={p.name: digest(p) for p in old.iterdir()})
    update(model, opt)
    restored = torch.nn.Linear(2, 1)
    restored.bias.requires_grad_(False)
    resumed = torch.optim.Adam([restored.weight], lr=3e-4)
    assert restore_completed(restored, resumed, frozen, out, recovery)[0] == 4
    update(restored, resumed)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert all(digest(out / n) == h for n, h in recovery["resume_files"].items())
    with pytest.raises(ValueError, match="frozen WM2"):
        restore_completed(restored, resumed, {"bias": frozen["bias"] + 1}, bad, recovery)
    assert not list(bad.iterdir())


def test_recorded_ctrl0_direct_row_reproduces_old_error_then_binds_feedback(tmp_path, monkeypatch):
    import wms.control.entrant as entrant
    from wms.data.action_timeline import CONTROLLERS
    from tools.wmrequal.grouped_feedback import feedback
    from tools.wmfixmerge.entrant_binding import bind_entrant_specs
    # grouped_feedback's sibling import needs the same path setup as Runtime.
    from tools.wmfixmerge import runtime  # noqa: F401
    checkpoint = tmp_path / "zero.pt"
    checkpoint.write_bytes(b"frozen-policy")
    spec = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint))
    specs = dict(ours_c1=spec, coopoff_c1=spec)
    timeline = {"controller": torch.full((1, 2, 1), CONTROLLERS.index("direct"))}
    with pytest.raises(ValueError, match="explicit feedback controller"):
        feedback(specs, ["ctrl0"], None, timeline, 1, "cpu")
    callback = object()
    monkeypatch.setattr(entrant, "policy_feedback", lambda *a, **kw: callback)
    bound = bind_entrant_specs(specs, [dict(arm="ctrl0", policy_sha256=spec["sha256"],
                                         controllers_used=["native", "direct"])])
    assert feedback(bound, ["ctrl0"], None, timeline, 1, "cpu").groups == [([0], callback)]
