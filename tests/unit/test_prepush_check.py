"""Tests for tools/prepush_check.py (the pre-push safety gate).

Every fake secret is assembled at run time from fragments so this file itself
contains nothing the gate would flag (see ``test_gate_passes_on_its_own_files``).
All repositories are throw-away git repos under ``tmp_path``.
"""
from __future__ import annotations

import json
import os
import shutil
import string
import subprocess
from pathlib import Path

import pytest

from tools import prepush_check as pc

ALNUM = string.ascii_letters + string.digits  # 62 chars; step 7 is coprime with 62


def tok(n: int, salt: int = 0) -> str:
    """Deterministic, varied (upper + lower + digits) fake secret body of length n."""
    return "".join(ALNUM[(i * 7 + salt) % len(ALNUM)] for i in range(n))


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def commit(repo: Path) -> None:
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-q", "-m", "snapshot")


@pytest.fixture
def make_repo(tmp_path):
    def make(files: dict[str, bytes | str]) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        git(repo, "init", "-q")
        write(repo, files)
        git(repo, "add", "-f", "-A")
        return repo

    return make


def write(repo: Path, files: dict[str, bytes | str]) -> None:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")


def run_json(capsys, repo: Path, *extra: str) -> tuple[int, dict]:
    capsys.readouterr()
    code = pc.main(["--repo", str(repo), "--json", *extra])
    out = capsys.readouterr().out
    return code, json.loads(out)


def rules_of(rep: dict, key: str = "blocking") -> set[str]:
    return {h["rule"] for h in rep[key]}


def hits(rep: dict, rule: str, key: str = "blocking") -> list[dict]:
    return [h for h in rep[key] if h["rule"] == rule]


# ---------------------------------------------------------------- clean repo
def test_clean_repo_passes(make_repo, capsys):
    repo = make_repo({"README.md": "hello\n", "src/a.py": "x = 1\n"})
    code, rep = run_json(capsys, repo)
    assert code == 0 and rep["passed"] is True
    assert rep["blocking"] == [] and rep["warnings"] == []
    assert rep["totals"]["files"] == 2
    assert rep["totals"]["bytes"] == len("hello\n") + len("x = 1\n")


def test_only_tracked_files_are_inspected(make_repo, capsys):
    repo = make_repo({"a.txt": "fine\n"})
    write(repo, {"untracked.txt": "api_key = '" + tok(24) + "'\n"})
    code, rep = run_json(capsys, repo)
    assert code == 0 and rep["totals"]["files"] == 1


# ------------------------------------------------------------ secret content
def _clash() -> str:
    return ("mixed-port: 7890\nproxies:\n  - name: n1\n    type: trojan\n"
            "    server: 203.0.113.5\n    port: 443\n    password: " + tok(16) + "\n")


# rule, text, 0-based line index (inside the snippet) that must be reported
SECRET_CASES = [
    ("SEC-PRIVKEY", "-----BEGIN " + "OPENSSH " + "PRIVATE KEY-----\n" + tok(40) + "\n", 0),
    ("SEC-PRIVKEY", "-----BEGIN " + "PRIVATE KEY-----\n" + tok(40) + "\n", 0),
    ("SEC-GITHUB-TOKEN", "t = '" + "gh" + "p_" + tok(36) + "'\n", 0),
    ("SEC-GITHUB-TOKEN", "t = '" + "github" + "_pat_" + tok(40) + "'\n", 0),
    ("SEC-GITHUB-TOKEN", "t = '" + "gh" + "o_" + tok(36, 3) + "'\n", 0),
    ("SEC-GITHUB-TOKEN", "t = '" + "gh" + "s_" + tok(36, 9) + "'\n", 0),
    ("SEC-ANTHROPIC-KEY", "k = '" + "sk-" + "ant-api03-" + tok(40) + "'\n", 0),
    ("SEC-OPENAI-KEY", "k = '" + "sk-" + tok(48) + "'\n", 0),
    ("SEC-OPENAI-KEY", "k = '" + "sk-" + "proj-" + tok(48) + "'\n", 0),
    ("SEC-AWS-KEY", "id = " + "AK" + "IA" + "Q3RT7YU9PL2MN4BV" + "\n", 0),
    ("SEC-GOOGLE-KEY", "g = " + "AI" + "za" + tok(35) + "\n", 0),
    ("SEC-GOOGLE-OAUTH", "a = " + "ya" + "29." + tok(40) + "\n", 0),
    ("SEC-GCP-SERVICE-ACCOUNT", '{"type": "service_' + 'account", "x": 1}\n', 0),
    ("SEC-SLACK-TOKEN", "s = " + "xo" + "xb-123456789012-" + tok(24) + "\n", 0),
    ("SEC-HF-TOKEN", "h = " + "h" + "f_" + tok(34) + "\n", 0),
    ("SEC-TENCENT-KEY", "t = " + "AK" + "ID" + tok(32) + "\n", 0),
    ("SEC-ZHIPU-KEY", "glm = " + "0123456789abcdef" * 2 + "." + tok(16) + "\n", 0),
    ("SEC-GENERIC-ASSIGN", 'api_key = "' + tok(24) + '"\n', 0),
    ("SEC-GENERIC-ASSIGN", '{"password": "' + tok(20) + '"}\n', 0),
    ("SEC-GENERIC-ASSIGN", "client_secret: '" + tok(20) + "'\n", 0),
    ("SEC-ENV-ASSIGN", "export MY_SERVICE_TOKEN=" + tok(24) + "\n", 0),
    ("SEC-ENV-ASSIGN", "DB_PASSWORD=" + tok(20) + "\n", 0),
    ("SEC-AUTH-HEADER", "Authorization: Bearer " + tok(30) + "\n", 0),
    ("SEC-RCLONE-CONFIG",
     "[gdrive]\ntype = drive\nscope = drive\ntoken = {\"access\": \"" + tok(30) + "\"}\n", 3),
    ("SEC-PROXY-URI", "link = " + "vm" + "ess://" + tok(40) + "\n", 0),
    ("SEC-PROXY-SUB", "https://sub.service.net/api/v1/client/" + "subscribe?token=" + tok(32) + "\n", 0),
    ("SEC-URL-TOKEN", "https://api.service.net/v1/data?token=" + tok(20) + "\n", 0),
    ("SEC-URL-USERINFO", "https://alice:" + tok(16) + "@proxy.service.net:8080\n", 0),
    ("SEC-PROXY-CONFIG", _clash(), 6),
    ("SEC-SSHPASS", "sshpass -p " + tok(12) + " ssh box\n", 0),
]


@pytest.fixture(scope="module")
def secret_repo(tmp_path_factory):
    """One repo holding every secret case as cases/case_NN.txt (built once per module)."""
    mp = pytest.MonkeyPatch()
    mp.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    mp.setenv("GIT_CONFIG_NOSYSTEM", "1")
    try:
        repo = tmp_path_factory.mktemp("secrets") / "repo"
        repo.mkdir()
        git(repo, "init", "-q")
        write(repo, {f"cases/case_{i:02d}.txt": "line one\nline two\n" + c[1]
                     for i, c in enumerate(SECRET_CASES)})
        git(repo, "add", "-f", "-A")
        yield repo
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def secret_report(secret_repo):
    return pc.run_checks(secret_repo).to_json()


@pytest.mark.parametrize("index", range(len(SECRET_CASES)),
                         ids=[f"{c[0]}-{i}" for i, c in enumerate(SECRET_CASES)])
def test_secret_rule_blocks_with_line_number(secret_report, index):
    rule, _snippet, offset = SECRET_CASES[index]
    path = f"cases/case_{index:02d}.txt"
    assert secret_report["passed"] is False
    found = [h for h in hits(secret_report, rule) if h["path"] == path]
    assert found, f"{rule} not reported for {path}"
    assert 3 + offset in {h["line"] for h in found}


def test_secret_values_are_never_printed(secret_repo, capsys):
    # tok(12) is a prefix of every tok(n) used above, so it detects any leaked secret body
    forbidden = [tok(12), "0123456789abcdef0123456789abcdef"]
    for extra in ([], ["--json"], ["--context"], ["--context", "--json"], ["--max-list", "0"]):
        capsys.readouterr()
        assert pc.main(["--repo", str(secret_repo), *extra]) == 1
        captured = capsys.readouterr()
        for text in (captured.out, captured.err):
            assert not any(f in text for f in forbidden), extra


def test_context_is_masked(make_repo, capsys):
    secret = tok(24)
    other = tok(30, 5)
    repo = make_repo({"cfg.txt": f'note = "{other}"; api_key = "{secret}"\n'})
    capsys.readouterr()
    assert pc.main(["--repo", str(repo), "--context"]) == 1
    out = capsys.readouterr().out
    assert "<MASKED:SEC-GENERIC-ASSIGN>" in out
    assert secret not in out and other not in out
    assert "<tok>" in out


def test_placeholders_and_prose_are_not_flagged(make_repo, capsys):
    text = "\n".join([
        'api_key = "<your-api-key-goes-here>"',
        'token = "${GITHUB_TOKEN_VALUE_HERE}"',
        'password = "' + "x" * 20 + '"',
        'secret = "this is a long sentence with spaces in it"',
        "export OPENAI_API_KEY=$OPENAI_API_KEY_FROM_ENV",
        'url = "https://user:password@localhost:5432/db"',
        "https://example.invalid/page?token=<TOKEN>",
        "AK" + "IAIOSFODNN7EXAMPLE",
        "tokens = '" + tok(24) + "'",   # key is 'tokens', not 'token'
        "max_token_len = 4096",
        "Authorization: Bearer ${TOKEN_FROM_ENV_VARIABLE}",
        "",
    ])
    code, rep = run_json(capsys, make_repo({"docs/readme.txt": text}))
    assert code == 0, rep["blocking"]
    assert rep["blocking"] == []


def test_rclone_block_needs_a_secret_line(make_repo, capsys):
    ini = "[gdrive]\ntype = drive\nscope = drive\n"
    code, rep = run_json(capsys, make_repo({"rclone_notes.txt": ini}))
    assert code == 0


def test_binary_files_are_not_content_scanned(make_repo, capsys):
    body = b"\x00\x01\x02" + ("api_key = '" + tok(24) + "'\n").encode()
    code, rep = run_json(capsys, make_repo({"asset/blob.dat": body}))
    assert code == 0
    assert rep["totals"]["skipped_binary"] == 1 and rep["totals"]["content_scanned"] == 0


def test_many_hits_in_one_file_are_folded(make_repo, capsys):
    lines = "".join("t%d = '%s'\n" % (i, "gh" + "p_" + tok(36, i)) for i in range(60))
    code, rep = run_json(capsys, make_repo({"many.txt": lines}))
    assert code == 1
    found = hits(rep, "SEC-GITHUB-TOKEN")
    assert sum(h["count"] for h in found) == 60
    assert any(h["line"] is None and h["count"] == 10 for h in found)
    row = next(r for r in rep["summary"] if r["rule"] == "SEC-GITHUB-TOKEN")
    assert row["block"] == 60


# ------------------------------------------------------------------- names
def test_env_and_credential_file_names(make_repo, capsys):
    files = {".env": "A=1\n", "cfg/.env.local": "A=1\n", "prod.env": "A=1\n",
             ".env.example": "A=1\n", "rclone.conf": "# empty\n", "keys/id_rsa": "x\n",
             "keys/id_rsa.pub": "x\n", "ca.pem": "x\n", ".ssh/config": "x\n",
             "service_account_utils.py": "x = 1\n", "client_secret_abc.json": "{}\n"}
    code, rep = run_json(capsys, make_repo(files))
    assert code == 1
    flagged = {h["path"] for h in rep["blocking"] if h["rule"].startswith("NAME-")}
    assert flagged == {".env", "cfg/.env.local", "prod.env", "rclone.conf", "keys/id_rsa",
                       "ca.pem", ".ssh/config", "client_secret_abc.json"}


# -------------------------------------------------------------------- paths
def test_forbidden_directories(make_repo, capsys):
    files = {"sealed/a.txt": "x\n", "x/sealed/b.txt": "x\n", "data_cache/k/v.txt": "x\n",
             "ok/sealedness.txt": "x\n"}
    code, rep = run_json(capsys, make_repo(files))
    assert code == 1
    assert {h["path"] for h in hits(rep, "PATH-SEALED")} == {"sealed/a.txt", "x/sealed/b.txt"}
    assert {h["path"] for h in hits(rep, "PATH-DATA-CACHE")} == {"data_cache/k/v.txt"}
    assert "ok/sealedness.txt" not in {h["path"] for h in rep["blocking"]}


def test_runs_only_allows_small_text_records(make_repo):
    small_text = ["NOTE.md", "r.json", "t.txt", "c.toml", "d.csv", "run.log", "rows.jsonl",
                  "job.out", "step.time", "go.ps1", "tool.py", "go.sh"]
    files = {f"runs/active/{n}": "x = 1\n" for n in small_text}
    files["runs/x.log"] = "log\n"
    too_big = {"runs/active/big.json": "x" * 200, "runs/active/big.log": "x" * 200,
               "runs/active/big.py": "x" * 200}
    not_text = {"runs/active/blob.dat": "x\n", "runs/active/img.png": "x\n",
                "runs/active/noext": "x\n", "runs/active/tool.pyc": "x\n",
                "runs/model.bin": "x\n"}
    repo = make_repo({**files, **too_big, **not_text})
    rep = pc.run_checks(repo, limits=pc.Limits(runs_text_max=100))
    assert {h.path for h in rep.blocking if h.rule == "PATH-RUNS"} == set(too_big) | set(not_text)
    # the text-like extensions are allowed up to the size limit, and only under runs/
    rep = pc.run_checks(repo, limits=pc.Limits(runs_text_max=100))
    assert not {h.path for h in rep.hits if h.rule == "PATH-RUNS"} & set(files)


def test_binary_extensions(make_repo):
    names = ["m.pt", "m.pth", "m.ckpt", "m.pkl", "a.npy", "a.npz", "d.parquet", "z.zst", "z.zip",
             "t.tar", "t.tar.gz", "t.tar.zst", "t.tgz"]
    other = ["m.onnx", "w.safetensors", "d.h5"]
    ok = ["osm.net.xml.gz", "notes.txt", "guitar.py", "tarball.md", "t.tar.sha256"]
    repo = make_repo({n: "x\n" for n in names + other + ok})
    rep = pc.run_checks(repo)
    assert {h.path for h in rep.blocking if h.rule == "PATH-BINARY-EXT"} == set(names)
    assert {h.path for h in rep.warnings if h.rule == "PATH-BINARY-OTHER"} == set(other)
    assert not {h.path for h in rep.hits} & set(ok)


def test_other_agents_config(make_repo):
    bad = [".cursor/rules.md", ".zcode/x.json", ".zcode-local/y.json", ".codex/config.toml",
           ".opencode/z.json", ".kimi/a.md", "sub/.cursor/b.md", "opencode.json"]
    repo = make_repo({n: "{}\n" for n in bad + ["docs/cursor.md", "AGENTS.md"]})
    rep = pc.run_checks(repo)
    assert {h.path for h in rep.blocking if h.rule == "PATH-AGENT-CONFIG"} == set(bad)


# ----------------------------------------------------------- name-rule words
def test_name_rule_block_versus_warn(make_repo):
    warn = ["tests/test_holdout_guard.py", "tools/holdout_guard.py", "protocols/chunk5_seal.json",
            "docs/INC1_notes.md", "archive/old/J4B/readme.md", "src/heldout_eval.py",
            "traffic_shepherd/cal_split.py",
            # any .py is code, whatever the directory
            "output/wd01/heldout/e.py", "scripts/j4b_run.py", "output/wd01/inc1/gates.py",
            "x/y/CHUNK5_tool.py"]
    block = ["data/holdout_set.csv", "output/wd01/inc1/lock.json", "tests/holdout_data.npz",
             "scripts/j4b_run.sh", "docs/chunk5_world.txt", "output/wd01/inc1/RUNBOOK.md",
             "scripts/holdout_notes.json", "_out/lpwo/holdout_eval.json"]
    repo = make_repo({n: "x = 1\n" for n in warn + block})
    rep = pc.run_checks(repo)
    got_warn = {h.path for h in rep.hits if h.rule == "NAME-RULE" and h.severity == pc.WARN}
    got_block = {h.path for h in rep.hits if h.rule == "NAME-RULE" and h.severity == pc.BLOCK}
    assert got_warn == set(warn)
    assert got_block == set(block)


def test_never_open_files_are_not_read(make_repo, monkeypatch):
    secret = "api_key = '" + tok(24) + "'\n"
    names = ["tests/holdout_x.py", "sealed/notes.txt", "data/cal_split.csv", "a/J4B/x.md",
             "chunk5/inc1_a.json"]
    files = {n: secret for n in names}
    files["open/ok.txt"] = secret
    repo = make_repo(files)
    opened: list[str] = []
    real = pc._WorktreeReader.read

    def spy(self, entry, limit):
        opened.append(entry.path)
        return real(self, entry, limit)

    monkeypatch.setattr(pc._WorktreeReader, "read", spy)
    rep = pc.run_checks(repo)
    assert opened == ["open/ok.txt"]
    assert rep.skipped_never_open == len(names)
    assert {h.path for h in rep.hits if h.rule == "SEC-GENERIC-ASSIGN"} == {"open/ok.txt"}


# --------------------------------------------------------------------- sizes
def test_default_limits_are_20_50_100_mib():
    lim = pc.Limits()
    assert (lim.warn, lim.block, lim.hard) == (20 * 1048576, 50 * 1048576, 100 * 1048576)


def test_size_thresholds_and_total(make_repo):
    lim = pc.Limits(warn=100, block=1000, hard=5000, total_warn=7000)
    files = {"ok.txt": b"a" * 100, "warn.txt": b"a" * 101, "block.txt": b"a" * 1001,
             "hard.txt": b"a" * 5001, "edge.txt": b"a" * 1000}
    repo = make_repo(files)
    rep = pc.run_checks(repo, limits=lim)
    by_rule = {h.rule: h.path for h in rep.hits}
    assert by_rule["SIZE-BLOCK"] == "block.txt"
    assert by_rule["SIZE-HARD"] == "hard.txt"
    assert {h.path for h in rep.hits if h.rule == "SIZE-WARN"} == {"warn.txt", "edge.txt"}
    assert {h.severity for h in rep.hits if h.rule == "SIZE-WARN"} == {pc.WARN}
    assert rep.total_bytes == sum(len(v) for v in files.values())
    assert any(h.rule == "SIZE-TOTAL" for h in rep.hits)
    assert not any(h.path == "ok.txt" for h in rep.hits)


def test_size_block_is_allowable_but_hard_limit_is_not(make_repo):
    lim = pc.Limits(warn=100, block=1000, hard=5000)
    repo = make_repo({"block.txt": b"a" * 1001, "hard.txt": b"a" * 5001})
    allow = ("block.txt SIZE-BLOCK # reviewed: archived table, needed\n"
             "hard.txt SIZE-HARD # must have no effect\n")
    rep = pc.run_checks(repo, allow_text=allow, limits=lim)
    assert [h.path for h in rep.blocking] == ["hard.txt"]
    assert [h.path for h in rep.allowed] == ["block.txt"]
    assert rep.unused_allow == ["hard.txt SIZE-HARD"]


# ---------------------------------------------------------------- allow-list
def test_allowlist_path_and_rule(make_repo, capsys):
    text = 'api_key = "' + tok(24) + '"\n'
    repo = make_repo({"docs/sample.txt": text, "other.txt": text})
    code, rep = run_json(capsys, repo)
    assert code == 1 and len(hits(rep, "SEC-GENERIC-ASSIGN")) == 2
    allow = repo / "allow.txt"
    allow.write_text("docs/sample.txt SEC-GENERIC-ASSIGN # reviewed: synthetic example\n",
                     encoding="utf-8")
    code, rep = run_json(capsys, repo, "--allow", str(allow))
    assert code == 1
    assert [h["path"] for h in hits(rep, "SEC-GENERIC-ASSIGN")] == ["other.txt"]
    assert [h["path"] for h in rep["allowed"]] == ["docs/sample.txt"]
    allow.write_text("docs/*.txt SEC-GENERIC-ASSIGN # g\nother.txt SEC-GENERIC-ASSIGN # g\n",
                     encoding="utf-8")
    code, rep = run_json(capsys, repo, "--allow", str(allow))
    assert code == 0 and rep["blocking"] == [] and len(rep["allowed"]) == 2
    row = next(r for r in rep["summary"] if r["rule"] == "SEC-GENERIC-ASSIGN")
    assert row["block"] == 0 and row["allowed"] == 2


def test_allowlist_is_rule_specific(make_repo):
    repo = make_repo({"x.txt": 'api_key = "' + tok(24) + '"\n'})
    rep = pc.run_checks(repo, allow_text="x.txt SEC-OPENAI-KEY # wrong rule\n")
    assert rep.blocking and rep.unused_allow == ["x.txt SEC-OPENAI-KEY"]


def test_allowlist_line_scope(make_repo):
    lines = ('api_key = "' + tok(24, 1) + '"\n' + "ok = 1\n" + 'api_key = "' + tok(24, 2) + '"\n')
    repo = make_repo({"x.txt": lines})
    rep = pc.run_checks(repo, allow_text="x.txt:1 SEC-GENERIC-ASSIGN # first line reviewed\n")
    assert [h.line for h in rep.blocking] == [3]
    assert [h.line for h in rep.allowed] == [1]


def test_allowlist_discovered_in_repo_and_validated(make_repo, capsys):
    text = 'api_key = "' + tok(24) + '"\n'
    allow = "docs/sample.txt SEC-GENERIC-ASSIGN # reviewed synthetic\n"
    repo = make_repo({"docs/sample.txt": text, pc.ALLOW_FILE_REL: allow})
    code, rep = run_json(capsys, repo)
    assert code == 0 and len(rep["allowed"]) == 1


@pytest.mark.parametrize("bad", [
    "docs/a.txt SEC-GENERIC-ASSIGN\n",              # no reason
    "docs/a.txt SEC-GENERIC-ASSIGN #   \n",         # empty reason
    "docs/a.txt NOT-A-RULE # reason\n",             # unknown rule
    "SEC-GENERIC-ASSIGN # only one field\n",        # malformed
])
def test_bad_allowlist_exits_2(make_repo, capsys, bad):
    repo = make_repo({"a.txt": "x\n"})
    allow = repo / "allow.txt"
    allow.write_text(bad, encoding="utf-8")
    capsys.readouterr()
    assert pc.main(["--repo", str(repo), "--allow", str(allow)]) == 2
    assert "allow-list" in capsys.readouterr().err


def test_allowlist_comments_and_blank_lines(make_repo):
    repo = make_repo({"a.txt": "x\n"})
    rep = pc.run_checks(repo, allow_text="# header\n\n   \n# more\n")
    assert rep.unused_allow == [] and not rep.hits


# -------------------------------------------------------------- misc / modes
def test_missing_tracked_file_is_a_warning(make_repo, capsys):
    repo = make_repo({"a.txt": "x\n", "gone.txt": "y\n"})
    (repo / "gone.txt").unlink()
    code, rep = run_json(capsys, repo)
    assert code == 0
    assert [(h["path"], h["rule"]) for h in rep["warnings"]] == [("gone.txt", "INFO-MISSING")]


def test_rev_mode_inspects_the_commit_not_the_worktree(make_repo, capsys):
    secret_line = 'api_key = "' + tok(24) + '"\n'
    repo = make_repo({"cfg.txt": secret_line, "tests/holdout_a.py": secret_line,
                      "ok/data.bin": b"\x00\x01" + secret_line.encode()})
    commit(repo)
    write(repo, {"cfg.txt": "clean\n"})
    git(repo, "add", "-f", "-A")
    code, rep = run_json(capsys, repo)
    assert code == 0 and rep["mode"] == "worktree"
    code, rep = run_json(capsys, repo, "--rev", "HEAD")
    assert code == 1 and rep["mode"] == "rev" and rep["rev"] == "HEAD"
    assert {h["path"] for h in hits(rep, "SEC-GENERIC-ASSIGN")} == {"cfg.txt"}
    assert rep["totals"]["skipped_binary"] == 1 and rep["totals"]["skipped_never_open"] == 1
    assert any(h["path"] == "tests/holdout_a.py" and h["rule"] == "NAME-RULE"
               for h in rep["warnings"])


def test_rev_mode_reads_allowlist_from_the_commit(make_repo, capsys):
    text = 'api_key = "' + tok(24) + '"\n'
    allow = "docs/sample.txt SEC-GENERIC-ASSIGN # reviewed synthetic\n"
    repo = make_repo({"docs/sample.txt": text, pc.ALLOW_FILE_REL: allow})
    commit(repo)
    code, rep = run_json(capsys, repo, "--rev", "HEAD")
    assert code == 0 and len(rep["allowed"]) == 1


def test_errors_exit_2(tmp_path, make_repo, capsys):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    capsys.readouterr()
    assert pc.main(["--repo", str(plain)]) == 2
    repo = make_repo({"a.txt": "x\n"})
    assert pc.main(["--repo", str(repo), "--rev", "no-such-rev"]) == 2
    assert "error" in capsys.readouterr().err


def test_json_and_text_summary_shape(make_repo, capsys):
    repo = make_repo({".env": "A=1\n", "tests/holdout_x.py": "x = 1\n", "m.pt": b"abc"})
    code, rep = run_json(capsys, repo)
    assert code == 1
    assert set(rep) >= {"passed", "totals", "summary", "blocking", "warnings", "allowed", "mode"}
    rows = {r["rule"]: r for r in rep["summary"]}
    assert rows["NAME-ENV-FILE"]["block"] == 1 and rows["PATH-BINARY-EXT"]["block"] == 1
    assert rows["NAME-RULE"]["warn"] == 1 and rows["NAME-RULE"]["block"] == 0
    capsys.readouterr()
    assert pc.main(["--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "RULE" in out and "BLOCK" in out and "WARN" in out and "RESULT: FAIL" in out
    assert "total size" in out and "tests/holdout_x.py" in out


def test_warnings_alone_do_not_fail(make_repo, capsys):
    repo = make_repo({"tests/holdout_x.py": "x = 1\n"})
    code, rep = run_json(capsys, repo)
    assert code == 0 and rep["passed"] is True and len(rep["warnings"]) == 1
    capsys.readouterr()
    assert pc.main(["--repo", str(repo)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out


def test_list_rules(capsys):
    assert pc.main(["--list-rules"]) == 0
    out = capsys.readouterr().out
    for rid in ("SIZE-BLOCK", "SEC-PRIVKEY", "NAME-RULE", "PATH-SEALED", "PATH-RUNS"):
        assert rid in out


# --------------------------------------------------------------- --extra-list
def _extra_repo(make_repo):
    repo = make_repo({"ok.txt": "fine\n"})
    secret = 'api_key = "' + tok(24) + '"\n'
    write(repo, {"new/a.txt": secret, "new/sub/b.txt": "clean\n", "runs/x.dat": "x = 1\n",
                 "runs/ok.log": "log\n", "tests/holdout_n.py": secret, "not_listed.txt": secret})
    lst = repo / "extra.txt"
    lst.write_text("# new files\n\nnew\nruns/x.dat\nruns/ok.log\ntests/holdout_n.py\nok.txt\n"
                   "./missing/f.txt\n",
                   encoding="utf-8")
    return repo, lst


def test_extra_list_is_reported_separately(make_repo, capsys):
    repo, lst = _extra_repo(make_repo)
    code, rep = run_json(capsys, repo, "--extra-list", str(lst))
    assert code == 1 and rep["passed"] is False and rep["passed_tracked"] is True
    assert rep["blocking"] == [] and rep["totals"]["files"] == 1
    ex = rep["extra"]
    assert ex["passed"] is False and ex["already_tracked_skipped"] == 1
    # a.txt, sub/b.txt, runs/x.dat, runs/ok.log, holdout_n.py, missing f.txt
    assert ex["totals"]["files"] == 6
    assert [(h["path"], h["line"]) for h in hits(ex, "SEC-GENERIC-ASSIGN")] == [("new/a.txt", 1)]
    assert [h["path"] for h in hits(ex, "PATH-RUNS")] == ["runs/x.dat"]  # small .log is allowed
    assert [(h["path"], h["rule"]) for h in ex["warnings"]] == [
        ("missing/f.txt", "INFO-MISSING"), ("tests/holdout_n.py", "NAME-RULE")]
    assert ex["totals"]["skipped_never_open"] == 1  # the holdout-named file was not read
    assert not any(h["path"] == "not_listed.txt" for h in ex["blocking"] + ex["warnings"])


def test_extra_list_text_output_and_exit_code(make_repo, capsys):
    repo, lst = _extra_repo(make_repo)
    capsys.readouterr()
    assert pc.main(["--repo", str(repo), "--extra-list", str(lst)]) == 1
    out = capsys.readouterr().out
    assert "EXTRA-LIST" in out and "new/a.txt:1" in out and "extra: 2 blocking" in out
    assert "tracked: 0 blocking" in out and tok(12) not in out


def test_extra_list_clean_passes_and_tracked_failure_still_fails(make_repo, capsys):
    repo = make_repo({"ok.txt": "fine\n"})
    write(repo, {"new/b.txt": "clean\n"})
    lst = repo / "extra.txt"
    lst.write_text("new/b.txt\n", encoding="utf-8")
    code, rep = run_json(capsys, repo, "--extra-list", str(lst))
    assert code == 0 and rep["passed"] and rep["extra"]["totals"]["files"] == 1
    write(repo, {".env": "A=1\n"})
    git(repo, "add", "-f", ".env")
    code, rep = run_json(capsys, repo, "--extra-list", str(lst))
    assert code == 1 and rep["passed_tracked"] is False and rep["extra"]["passed"] is True


def test_extra_list_uses_the_allow_list(make_repo):
    repo, _lst = _extra_repo(make_repo)
    allow = ("new/a.txt SEC-GENERIC-ASSIGN # reviewed synthetic\n"
             "runs/x.dat PATH-RUNS # reviewed\n"
             "stale.txt SEC-GENERIC-ASSIGN # matches nothing\n")
    rep = pc.run_checks(repo, allow_text=allow, extra_paths=["new", "runs/x.dat"])
    assert rep.extra is not None and rep.extra.blocking == [] and len(rep.extra.allowed) == 2
    assert rep.unused_allow == ["stale.txt SEC-GENERIC-ASSIGN"]


@pytest.mark.parametrize("bad", ["../outside.txt", "/abs/path.txt", "C:/Users/x/y.txt",
                                 "a/../../b.txt"])
def test_extra_list_rejects_non_relative_paths(make_repo, capsys, bad):
    repo = make_repo({"ok.txt": "fine\n"})
    lst = repo / "extra.txt"
    lst.write_text(bad + "\n", encoding="utf-8")
    capsys.readouterr()
    assert pc.main(["--repo", str(repo), "--extra-list", str(lst)]) == 2
    assert "repo-relative" in capsys.readouterr().err


def test_extra_list_cannot_be_combined_with_rev(make_repo, capsys):
    repo = make_repo({"ok.txt": "fine\n"})
    commit(repo)
    lst = repo / "extra.txt"
    lst.write_text("ok.txt\n", encoding="utf-8")
    capsys.readouterr()
    assert pc.main(["--repo", str(repo), "--rev", "HEAD", "--extra-list", str(lst)]) == 2


def test_gate_passes_on_its_own_files(make_repo):
    """The gate and its tests (which hold only fragments) must not trip the gate."""
    mine = {"tools/prepush_check.py": Path(pc.__file__), "tests/unit/test_prepush_check.py":
            Path(__file__)}
    allow = Path(pc.__file__).with_name("prepush_allow.txt")
    if allow.exists():
        mine[pc.ALLOW_FILE_REL] = allow
    repo = make_repo({})
    for rel, src in mine.items():
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    git(repo, "add", "-f", "-A")
    rep = pc.run_checks(repo, allow_text="")
    assert [(h.path, h.rule, h.line) for h in rep.hits] == []
