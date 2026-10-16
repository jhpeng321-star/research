#!/usr/bin/env python
"""Pre-push safety gate for the fresh single-commit private GitHub repository.

Inspects exactly the files that ``git ls-files`` lists (or, with ``--rev REV``,
the files of that commit) and exits non-zero on any blocking hit.  Stdlib only,
read-only (``ls-files`` / ``ls-tree`` / ``cat-file`` / ``rev-parse``).

Rule families (``--list-rules`` prints every id):

* SIZE-*   file size (warn > 20 MiB, block > 50 MiB, hard GitHub limit 100 MiB)
* SEC-*    secret scan of text content (keys, tokens, rclone blocks, proxy URLs)
* NAME-*   credential / .env file names, sealed-or-holdout style names
* PATH-*   forbidden locations (sealed/, data_cache/, runs/, binaries, agent config)

Secrets are never printed: a hit is reported as path, line number and rule id.
``--context`` additionally prints the surrounding line with the matched span
replaced by a mask and any other long opaque token replaced by ``<tok>``.

Files under ``sealed/`` and files whose path contains holdout / heldout /
cal_split / chunk5 / j4b / inc1 are never opened (reported by name only).

Reviewed false positives go in ``tools/prepush_allow.txt``::

    <path-or-glob>[:LINE]  <RULE-ID>  # mandatory reason

Usage::

    python tools/prepush_check.py                 # working tree, text summary
    python tools/prepush_check.py --json          # machine-readable, all hits
    python tools/prepush_check.py --rev HEAD      # a commit (e.g. the fresh single commit)
    python tools/prepush_check.py --repo DIR --context --max-list 0
    python tools/prepush_check.py --extra-list FILE   # also paths not yet tracked

Exit codes: 0 no blocking hit, 1 blocking hit(s), 2 usage / git error.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

MIB = 1024 * 1024
BLOCK, WARN = "block", "warn"
ALLOW_FILE_REL = "tools/prepush_allow.txt"
MAX_STORED = 50  # stored hits per (path, rule); the rest are folded into one summary hit
HEAD_BYTES = 8192

NAME_WORDS = ("holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
NAME_CODE_EXT = {".py", ".md", ".json"}
NAME_CODE_DIRS = {"protocols", "tests", "tools", "traffic_shepherd", "src", "docs", "archive"}
RUNS_TEXT_EXT = {".md", ".json", ".txt", ".toml", ".csv", ".log", ".jsonl", ".out", ".time", ".ps1", ".py", ".sh"}
BINARY_EXT = {".pt", ".pth", ".ckpt", ".pkl", ".npy", ".npz", ".parquet", ".zst", ".zip",
              ".tgz", ".tbz", ".tbz2", ".txz"}
# other model / array / archive formats: not on the block list, so WARN only
OTHER_BINARY_EXT = {".7z", ".rar", ".safetensors", ".onnx", ".h5", ".hdf5", ".pickle", ".joblib",
                    ".feather", ".arrow", ".bin"}
# .tar and compressed tars; checksum sidecars such as x.tar.sha256 are text and not matched
TAR_RX = re.compile(r"\.tar(\.(?:gz|bz2|xz|zst|zstd|lz4|lzma|z|br))?$")
AGENT_DIRS = {".cursor", ".codex", ".opencode", ".kimi"}
AGENT_FILES = {".cursorrules", ".cursorignore", "opencode.json", "opencode.jsonc"}
ENV_TEMPLATE_SUFFIX = (".example", ".sample", ".template", ".dist", ".tmpl", ".default")
CRED_NAMES = {
    "rclone.conf", ".rclone.conf", ".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials",
    ".htpasswd", "credentials.json", "secrets.json", "secrets.yaml", "secrets.yml",
    "secrets.toml",
}
CRED_SSH_PREFIX = ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")
CRED_JSON_PREFIX = ("client_secret", "service_account", "service-account")
CRED_EXT = {".pem", ".ppk", ".p12", ".pfx", ".jks", ".keystore", ".kdbx", ".key"}
CRED_DIRS = {".ssh", ".aws", ".gnupg", ".kube"}


@dataclass(frozen=True)
class Limits:
    warn: int = 20 * MIB
    block: int = 50 * MIB
    hard: int = 100 * MIB
    total_warn: int = 1024 * MIB
    runs_text_max: int = 1 * MIB


# ----------------------------------------------------------------------------
# Rule registry: id -> (default severity, description)
# ----------------------------------------------------------------------------
RULES: dict[str, tuple[str, str]] = {
    "SIZE-WARN": (WARN, "file larger than 20 MiB"),
    "SIZE-BLOCK": (BLOCK, "file larger than 50 MiB (allow-listable)"),
    "SIZE-HARD": (BLOCK, "file larger than 100 MiB, over the GitHub limit (not allow-listable)"),
    "SIZE-TOTAL": (WARN, "total tracked size above 1 GiB"),
    "SEC-PRIVKEY": (BLOCK, "PEM / OpenSSH / PGP / PuTTY / age private key material"),
    "SEC-GITHUB-TOKEN": (BLOCK, "GitHub token (ghp_/gho_/ghu_/ghs_/ghr_/github_pat_)"),
    "SEC-ANTHROPIC-KEY": (BLOCK, "Anthropic-style API key (sk-ant-)"),
    "SEC-OPENAI-KEY": (BLOCK, "OpenAI-style API key (sk-)"),
    "SEC-AWS-KEY": (BLOCK, "AWS access key id (AKIA / ASIA)"),
    "SEC-GOOGLE-KEY": (BLOCK, "Google API key (AIza)"),
    "SEC-GOOGLE-OAUTH": (BLOCK, "Google OAuth access / refresh token (ya29. / 1//)"),
    "SEC-GCP-SERVICE-ACCOUNT": (BLOCK, "Google service-account JSON key"),
    "SEC-SLACK-TOKEN": (BLOCK, "Slack token or webhook"),
    "SEC-HF-TOKEN": (BLOCK, "Hugging Face token (hf_)"),
    "SEC-TENCENT-KEY": (BLOCK, "Tencent Cloud secret id (AKID)"),
    "SEC-ZHIPU-KEY": (BLOCK, "Zhipu / BigModel API key shape (32 hex . 16 alnum)"),
    "SEC-GENERIC-ASSIGN": (BLOCK, "api_key / secret / token / password assigned a quoted literal"),
    "SEC-ENV-ASSIGN": (BLOCK, "NAME_KEY/SECRET/TOKEN/PASSWORD=<unquoted literal> env-style line"),
    "SEC-AUTH-HEADER": (BLOCK, "Authorization header carrying a literal credential"),
    "SEC-RCLONE-CONFIG": (BLOCK, "rclone config block with token / secret / pass lines"),
    "SEC-PROXY-URI": (BLOCK, "proxy share link (vmess:// vless:// trojan:// ss:// ...)"),
    "SEC-PROXY-SUB": (BLOCK, "Clash / proxy subscription URL"),
    "SEC-URL-TOKEN": (BLOCK, "URL whose query carries token= / key= / secret= / password="),
    "SEC-URL-USERINFO": (BLOCK, "URL with user:password@ credentials"),
    "SEC-PROXY-CONFIG": (BLOCK, "Clash-style proxy node or credential lines"),
    "SEC-SSHPASS": (BLOCK, "sshpass -p / plink -pw with a literal password"),
    "NAME-ENV-FILE": (BLOCK, ".env-style file (templates .example/.sample excluded)"),
    "NAME-CREDENTIAL-FILE": (BLOCK, "credential-looking file or directory name"),
    "NAME-RULE": (BLOCK, "path contains holdout/heldout/cal_split/chunk5/j4b/inc1; "
                         "WARN when any .py or code/protocol text, else BLOCK; never read"),
    "PATH-SEALED": (BLOCK, "anything under a sealed/ directory (never read)"),
    "PATH-DATA-CACHE": (BLOCK, "anything under a data_cache/ directory"),
    "PATH-RUNS": (BLOCK, "under runs/ and not a small (<=1 MiB) text record (md/json/txt/toml/csv/log/jsonl/out/time/ps1/py/sh)"),
    "PATH-BINARY-EXT": (BLOCK, "model / array / archive extension (pt pth ckpt pkl npy npz "
                               "parquet tar* zst zip ...)"),
    "PATH-BINARY-OTHER": (WARN, "other model / array / archive format (onnx h5 safetensors 7z ...)"),
    "PATH-AGENT-CONFIG": (BLOCK, "other agents' tool config (.cursor .zcode* .codex .opencode .kimi)"),
    "INFO-GITLINK": (WARN, "tracked submodule pointer (contents are not part of the push)"),
    "INFO-SYMLINK": (WARN, "tracked symlink (not followed, not scanned)"),
    "INFO-MISSING": (WARN, "listed by git ls-files but absent from the working tree"),
}
NOT_ALLOWABLE = {"SIZE-HARD"}


@dataclass(frozen=True)
class Entry:
    path: str
    mode: str
    sha: str
    size: int | None  # None = missing


@dataclass
class Hit:
    path: str
    rule: str
    severity: str
    line: int | None = None
    detail: str = ""
    count: int = 1
    allowable: bool = True
    context: str | None = None

    def to_json(self) -> dict:
        d = {"path": self.path, "rule": self.rule, "severity": self.severity,
             "line": self.line, "detail": self.detail, "count": self.count}
        if self.context is not None:
            d["context"] = self.context
        return d


def _hit(path: str, rule: str, detail: str = "", severity: str | None = None,
         line: int | None = None) -> Hit:
    return Hit(path=path, rule=rule, severity=severity or RULES[rule][0], line=line,
               detail=detail, allowable=rule not in NOT_ALLOWABLE)


# ----------------------------------------------------------------------------
# Path / name rules (never touch file content)
# ----------------------------------------------------------------------------
def _ext(name: str) -> str:
    return os.path.splitext(name)[1]


def name_rule_words(path: str) -> list[str]:
    low = path.lower()
    return [w for w in NAME_WORDS if w in low]


def is_never_open(path: str) -> bool:
    """sealed/ content and holdout-style names are reported by name, never read."""
    parts = path.lower().split("/")
    return "sealed" in parts[:-1] or bool(name_rule_words(path))


def path_hits(e: Entry, lim: Limits) -> list[Hit]:
    p = e.path
    parts = p.lower().split("/")
    name, dirs = parts[-1], parts[:-1]
    ext = _ext(name)
    size = e.size or 0
    out: list[Hit] = []

    if "sealed" in dirs:
        out.append(_hit(p, "PATH-SEALED", "under sealed/; content not read"))
    if "data_cache" in dirs:
        out.append(_hit(p, "PATH-DATA-CACHE", "under data_cache/"))
    if "runs" in dirs:
        if ext not in RUNS_TEXT_EXT:
            out.append(_hit(p, "PATH-RUNS", f"under runs/, extension '{ext or '(none)'}' "
                                            "is not a small text record"))
        elif size > lim.runs_text_max:
            out.append(_hit(p, "PATH-RUNS", f"under runs/, text record {size} bytes exceeds "
                                            f"{lim.runs_text_max}"))
    if ext in BINARY_EXT or TAR_RX.search(name):
        out.append(_hit(p, "PATH-BINARY-EXT", f"extension '{ext}'"))
    elif ext in OTHER_BINARY_EXT:
        out.append(_hit(p, "PATH-BINARY-OTHER", f"extension '{ext}'"))
    if (any(d in AGENT_DIRS or d.startswith(".zcode") for d in dirs)
            or name in AGENT_FILES or name.startswith(".zcode") or name in AGENT_DIRS):
        out.append(_hit(p, "PATH-AGENT-CONFIG", "another agent's tool configuration"))

    is_env = (name in (".env", ".envrc") or name.startswith(".env.") or name.endswith(".env"))
    if is_env and not name.endswith(ENV_TEMPLATE_SUFFIX):
        out.append(_hit(p, "NAME-ENV-FILE", ".env-style file"))
    cred = (name in CRED_NAMES or ext in CRED_EXT or any(d in CRED_DIRS for d in dirs)
            or (name.startswith(CRED_SSH_PREFIX) and not name.endswith(".pub"))
            or (name.startswith(CRED_JSON_PREFIX) and ext == ".json"))
    if cred:
        out.append(_hit(p, "NAME-CREDENTIAL-FILE", "credential-looking file name or directory"))

    words = name_rule_words(p)
    if words:
        is_code = ext == ".py" or (ext in NAME_CODE_EXT and parts[0] in NAME_CODE_DIRS)
        sev = WARN if is_code else BLOCK
        kind = "code/protocol text -> WARN" if is_code else "not code/protocol text -> BLOCK"
        out.append(_hit(p, "NAME-RULE", f"name matches {','.join(words)}; {kind}; content not read",
                        severity=sev))
    return out


# ----------------------------------------------------------------------------
# Content rules
# ----------------------------------------------------------------------------
_PLACEHOLDER_RX = re.compile(
    rb"(?is)^(?:[x*.\-_#0]+|\$.*|<.*>|%.*|\{.*|\[.*|.*(?:example|placeholder|changeme|change-me"
    rb"|redacted|dummy|sample|xxxx|\*\*\*\*|<[^>]*>|\{\{|\$\{|os\.environ|getenv|your[_-]?"
    rb"(?:token|key|secret|password|api)).*)$")
_PW_WORDS = {b"pass", b"password", b"passwd", b"pwd", b"secret", b"token", b"user", b"username",
             b"changeme", b"mypassword", b"yourpassword", b"p4ssw0rd", b"admin", b"root"}


def is_placeholder(value: bytes) -> bool:
    v = value.strip(b"'\"")
    return (not v or len(set(v)) <= 4 or v.lower() in _PW_WORDS
            or bool(_PLACEHOLDER_RX.match(v)))


Span = tuple[int, int]
Finder = Callable[[bytes, bytes], Iterator[Span]]  # (data, data.lower()) -> spans


def _value_span(m: re.Match) -> Span:
    gi = m.re.groupindex
    if "v" in gi and m.start("v") >= 0:
        return m.start("v"), m.end("v")
    return m.start(), m.end()


def _rx(pattern: str, flags: int = 0, *, check_placeholder: bool = False) -> Finder:
    rx = re.compile(pattern.encode("ascii"), flags)

    def find(data: bytes, lower: bytes) -> Iterator[Span]:
        for m in rx.finditer(data):
            s, e = _value_span(m)
            if check_placeholder and is_placeholder(data[s:e]):
                continue
            yield s, e

    return find


_RC_SECTION = re.compile(rb"(?m)^[ \t]*\[[A-Za-z0-9_.@ -]+\][ \t]*\r?$")
_RC_TYPE = re.compile(rb"(?im)^[ \t]*type[ \t]*=[ \t]*[a-z0-9-]+[ \t]*\r?$")
_RC_SECRET = re.compile(
    rb"(?im)^[ \t]*(?:token|client_secret|refresh_token|access_token|secret_access_key"
    rb"|access_key_id|service_account_credentials|pass|password|api_key)[ \t]*=[ \t]*"
    rb"(?P<v>\S[^\r\n]*)$")


def _find_rclone(data: bytes, lower: bytes) -> Iterator[Span]:
    if len(data) > 65536 or not (_RC_SECTION.search(data) and _RC_TYPE.search(data)):
        return
    for m in _RC_SECRET.finditer(data):
        v = m.group("v").strip()
        if (v.startswith(b"{") and b":" in v) or not is_placeholder(v):  # JSON token blob
            yield m.start("v"), m.end("v")


_CLASH_MARK = re.compile(
    rb"(?im)^\s*(?:mixed-port|socks-port|redir-port|external-controller|proxies|proxy-groups"
    rb"|proxy-providers|rule-providers)\s*:")
_CLASH_TYPE = re.compile(
    rb"(?im)^[ \t]*-?[ \t]*(?:\{[^\n]*?)?\btype[ \t]*:[ \t]*(?P<v>ss|ssr|vmess|vless|trojan"
    rb"|hysteria2?|tuic|wireguard)\b")
_CLASH_SERVER = re.compile(rb"(?im)^[ \t]*-?[ \t]*(?:\{[^\n]*?)?\bserver[ \t]*:[ \t]*\S")
_CLASH_CRED = re.compile(
    rb"(?im)^[ \t]*-?[ \t]*(?:password|uuid|secret|auth-str|psk|private-key)[ \t]*:[ \t]*"
    rb"['\"]?(?P<v>[^\s'\"#]{6,})")


def _find_clash(data: bytes, lower: bytes) -> Iterator[Span]:
    if len(data) > 2 * MIB or not (_CLASH_MARK.search(data) or (_CLASH_TYPE.search(data) and _CLASH_SERVER.search(data))):
        return
    for m in _CLASH_CRED.finditer(data):
        if not is_placeholder(m.group("v")):
            yield m.start("v"), m.end("v")
    if _CLASH_SERVER.search(data):
        for m in _CLASH_TYPE.finditer(data):
            yield m.start("v"), m.end("v")


_GEN_KEYS = (b"api_key", b"api-key", b"apikey", b"secret", b"token", b"password", b"passwd")
_GEN_TAIL = re.compile(rb"""['"]?[ \t]*[:=][ \t]*['"](?P<v>[^'"\s]{16,})['"]""")


def _find_generic(data: bytes, lower: bytes) -> Iterator[Span]:
    """api_key / secret / token / password followed by = or : and a quoted 16+ char literal."""
    for kw in _GEN_KEYS:
        pos = lower.find(kw)
        while pos >= 0:
            m = _GEN_TAIL.match(data, pos + len(kw))
            if m and not is_placeholder(m.group("v")):
                yield m.span("v")
            pos = lower.find(kw, pos + 1)


_ENV_WORDS = (b"KEY", b"SECRET", b"TOKEN", b"PASSWORD", b"PASSWD", b"PASSPHRASE")
_ENV_LINE = re.compile(
    rb"[ \t]*(?:export[ \t]+|set[ \t]+)?[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD"
    rb"|PASSPHRASE)[ \t]*=[ \t]*(?P<v>[^\s'\"#$<{%`\(][^\s#'\"]{15,})")


def _find_env_assign(data: bytes, lower: bytes) -> Iterator[Span]:
    """NAME_KEY / _SECRET / _TOKEN / _PASSWORD = unquoted literal at the start of a line."""
    seen: set[int] = set()
    for w in _ENV_WORDS:
        pos = data.find(w)
        while pos >= 0:
            lo = max(0, pos - 300)
            nl = data.rfind(b"\n", lo, pos)
            if nl >= 0 or lo == 0:
                ls = nl + 1
                if ls not in seen:
                    seen.add(ls)
                    m = _ENV_LINE.match(data, ls)
                    if m and not is_placeholder(m.group("v")):
                        yield m.span("v")
            pos = data.find(w, pos + 1)


@dataclass(frozen=True)
class ContentRule:
    rule: str
    prefilter: tuple[bytes, ...]  # lowercase literals; rule runs only if one is present
    find: Finder


_NB = r"(?<![A-Za-z0-9_-])"  # not inside a longer token
CONTENT_RULES: list[ContentRule] = [
    ContentRule("SEC-PRIVKEY", (b"private key", b"putty-user-key", b"age-secret-key"), _rx(
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|PuTTY-User-Key-File-\d"
        r"|AGE-SECRET-KEY-1[0-9A-Z]{50,}")),
    ContentRule("SEC-GITHUB-TOKEN",
                (b"ghp_", b"gho_", b"ghu_", b"ghs_", b"ghr_", b"github_pat_"), _rx(
        _NB + r"(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})",
        check_placeholder=True)),
    ContentRule("SEC-ANTHROPIC-KEY", (b"sk-ant-",), _rx(
        _NB + r"sk-ant-[A-Za-z0-9_-]{20,}", check_placeholder=True)),
    ContentRule("SEC-OPENAI-KEY", (b"sk-",), _rx(
        _NB + r"sk-(?!ant-)(?:proj-|svcacct-|admin-)?(?=[A-Za-z0-9_-]*[0-9A-Z])"
        r"[A-Za-z0-9_-]{20,}", check_placeholder=True)),
    ContentRule("SEC-AWS-KEY", (b"akia", b"asia"), _rx(
        r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])", check_placeholder=True)),
    ContentRule("SEC-GOOGLE-KEY", (b"aiza",), _rx(
        _NB + r"AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])", check_placeholder=True)),
    ContentRule("SEC-GOOGLE-OAUTH", (b"ya29.", b"1//"), _rx(
        r"(?<![A-Za-z0-9_-])ya29\.[0-9A-Za-z_-]{20,}|(?<![A-Za-z0-9/_:.-])1//[0-9A-Za-z_-]{40,}",
        check_placeholder=True)),
    ContentRule("SEC-GCP-SERVICE-ACCOUNT", (b"service_account", b"private_key_id"), _rx(
        r"\"type\"[ \t]*:[ \t]*\"service_account\"|\"private_key_id\"[ \t]*:[ \t]*\"[0-9a-f]{20,}\"")),
    ContentRule("SEC-SLACK-TOKEN", (b"xox", b"xapp-", b"hooks.slack.com"), _rx(
        r"xox[bpoars]-[0-9A-Za-z-]{10,}|xapp-\d-[A-Z0-9]+-\d+-[a-f0-9]+"
        r"|hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+",
        check_placeholder=True)),
    ContentRule("SEC-HF-TOKEN", (b"hf_",), _rx(
        r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{34,}", check_placeholder=True)),
    ContentRule("SEC-TENCENT-KEY", (b"akid",), _rx(
        r"(?<![A-Za-z0-9])AKID[A-Za-z0-9]{32}(?![A-Za-z0-9])", check_placeholder=True)),
    ContentRule("SEC-ZHIPU-KEY", (b"bigmodel", b"zhipu", b"glm", b"api_key", b"api-key",
                                  b"apikey", b"bearer", b"authorization"), _rx(
        r"(?<![0-9A-Za-z.])[0-9a-f]{32}\.[A-Za-z0-9]{16}(?![0-9A-Za-z])", check_placeholder=True)),
    ContentRule("SEC-GENERIC-ASSIGN", (b"key", b"secret", b"token", b"passw"), _find_generic),
    ContentRule("SEC-ENV-ASSIGN", (b"key", b"secret", b"token", b"passw"), _find_env_assign),
    ContentRule("SEC-AUTH-HEADER", (b"authorization",), _rx(
        r"authorization['\"]?[ \t]*[:=][ \t]*['\"]?(?:bearer|basic|token)[ \t]+"
        r"(?P<v>[A-Za-z0-9._~+/=-]{20,})", re.I, check_placeholder=True)),
    ContentRule("SEC-RCLONE-CONFIG", (b"type", b"token", b"pass", b"secret"), _find_rclone),
    ContentRule("SEC-PROXY-URI", (b"vmess://", b"vless://", b"trojan://", b"ss://", b"ssr://",
                                  b"hysteria", b"hy2://", b"tuic://", b"wireguard://"), _rx(
        r"(?<![A-Za-z0-9])(?:vmess|vless|trojan|ssr?|hysteria2?|hy2|tuic|wireguard)://"
        r"[A-Za-z0-9+/=_%@:.?&#~-]{16,}", re.I, check_placeholder=True)),
    ContentRule("SEC-PROXY-SUB", (b"http",), _rx(
        r"https?://[^\s'\"<>`]*(?:/api/v1/client/subscribe|/sub(?:scribe)?\?"
        r"|[?&]flag=(?:clash|meta|v2ray)|[?&]target=(?:clash|clashr|v2ray|ss|ssr|surge|quanx|loon"
        r"|singbox)|[?&]clash=\d)", re.I)),
    ContentRule("SEC-URL-TOKEN", (b"http",), _rx(
        r"https?://[^\s'\"<>`]{1,300}?[?&](?:token|key|secret|password|passwd|pwd|auth"
        r"|access_token|apikey|api_key|signature|sig|passkey|sessionid)="
        r"(?P<v>[^&\s'\"<>#`]{8,})", re.I, check_placeholder=True)),
    ContentRule("SEC-URL-USERINFO", (b"://",), _rx(
        r"[A-Za-z][A-Za-z0-9+.-]{1,15}://[^\s/:@'\"<>{}$%\\]{0,64}:"
        r"(?P<v>[^\s/@'\"<>{}$\\]{3,128})@[A-Za-z0-9_.\[-]+", check_placeholder=True)),
    ContentRule("SEC-PROXY-CONFIG", (b"server", b"proxies", b"proxy-groups", b"external-controller",
                                     b"mixed-port"), _find_clash),
    ContentRule("SEC-SSHPASS", (b"sshpass", b"plink"), _rx(
        r"(?:sshpass[ \t]+-p|plink\b[^\n]*[ \t]-pw)[ \t]*['\"]?(?P<v>[^\s'\"$<{%]{3,})",
        re.I, check_placeholder=True)),
]
assert all(c.rule in RULES for c in CONTENT_RULES)

_TOK_RX = re.compile(rb"[A-Za-z0-9+/_=.~-]{20,}")


def mask_context(data: bytes, start: int, end: int, rule: str) -> str:
    """The match line with the matched span masked and long opaque tokens removed."""
    ls = data.rfind(b"\n", 0, start) + 1
    le = data.find(b"\n", end)
    le = len(data) if le < 0 else le
    left = _TOK_RX.sub(b"<tok>", data[max(ls, start - 80):start])
    right = _TOK_RX.sub(b"<tok>", data[end:min(le, end + 80)])
    text = left + b"<MASKED:" + rule.encode() + b">" + right
    return text.decode("ascii", "replace").strip()[:240]


def scan_content(path: str, data: bytes, want_context: bool) -> list[Hit]:
    """All content hits for one text file (before allow-listing)."""
    lower = data.lower()
    found: dict[str, list[Span]] = {}
    for cr in CONTENT_RULES:
        if not any(lit in lower for lit in cr.prefilter):
            continue
        spans = list(cr.find(data, lower))
        if spans:
            found[cr.rule] = spans
    del lower
    hits: list[Hit] = []
    overflow: Counter = Counter()
    for rule, spans in found.items():
        spans.sort()
        seen: set[int] = set()
        pos, line = 0, 1
        stored = 0
        for s, e in spans:
            line += data.count(b"\n", pos, s)
            pos = s
            if line in seen:
                continue
            seen.add(line)
            if stored >= MAX_STORED:
                overflow[rule] += 1
                continue
            stored += 1
            h = _hit(path, rule, line=line)
            if want_context:
                h.context = mask_context(data, s, e, rule)
            hits.append(h)
        if overflow[rule]:
            hits.append(_hit(path, rule, f"+{overflow[rule]} more matching lines in this file "
                                         "(not listed)", line=None))
            hits[-1].count = overflow[rule]
    return hits


# ----------------------------------------------------------------------------
# Git access
# ----------------------------------------------------------------------------
class GitError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> bytes:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=env)
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args[:2])} failed: {r.stderr.decode('utf-8', 'replace')[:300]}")
    return r.stdout


def _fs(p: Path) -> str:
    s = str(p)
    if os.name == "nt" and len(s) > 240 and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + os.path.abspath(s)
    return s


def list_entries(repo: Path, rev: str | None) -> list[Entry]:
    entries: dict[str, Entry] = {}
    if rev is None:
        raw = _git(repo, "ls-files", "-s", "-z")
        for rec in raw.split(b"\0"):
            if not rec:
                continue
            meta, _, bpath = rec.partition(b"\t")
            mode, sha, _stage = meta.decode().split()
            path = os.fsdecode(bpath)
            if path in entries:
                continue
            try:
                size: int | None = os.lstat(_fs(repo / path)).st_size
            except OSError:
                size = None
            entries[path] = Entry(path, mode, sha, size)
    else:
        raw = _git(repo, "ls-tree", "-r", "-l", "-z", rev)
        for rec in raw.split(b"\0"):
            if not rec:
                continue
            meta, _, bpath = rec.partition(b"\t")
            mode, _typ, sha, size_s = meta.decode().split()
            path = os.fsdecode(bpath)
            entries[path] = Entry(path, mode, sha, int(size_s) if size_s.isdigit() else 0)
    return list(entries.values())


class _WorktreeReader:
    def __init__(self, repo: Path):
        self.repo = repo

    def read(self, e: Entry, limit: int) -> tuple[str, bytes]:
        if (e.size or 0) > limit:
            return "large", b""
        try:
            with open(_fs(self.repo / e.path), "rb") as fh:
                head = fh.read(HEAD_BYTES)
                if b"\0" in head:
                    return "binary", b""
                data = head + fh.read()  # size <= limit; read(limit) would pre-allocate it
        except OSError:
            return "missing", b""
        return ("large", b"") if len(data) > limit else ("text", data)

    def read_path(self, rel: str) -> bytes | None:
        try:
            return (self.repo / rel).read_bytes()
        except OSError:
            return None

    def close(self) -> None:
        pass


class _RevReader:
    def __init__(self, repo: Path, rev: str):
        self.repo, self.rev = repo, rev
        self.proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self, e: Entry, limit: int) -> tuple[str, bytes]:
        if (e.size or 0) > limit:
            return "large", b""
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(e.sha.encode() + b"\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            return "missing", b""
        data = self.proc.stdout.read(int(header[2]))
        self.proc.stdout.read(1)
        return ("binary", b"") if b"\0" in data[:HEAD_BYTES] else ("text", data)

    def read_path(self, rel: str) -> bytes | None:
        try:
            return _git(self.repo, "cat-file", "blob", f"{self.rev}:{rel}")
        except GitError:
            return None

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=30)


# ----------------------------------------------------------------------------
# Allow-list
# ----------------------------------------------------------------------------
@dataclass
class AllowEntry:
    pattern: str
    line: int | None
    rule: str
    reason: str
    lineno: int
    used: int = 0

    def matches(self, h: Hit) -> bool:
        if h.rule != self.rule or not h.allowable:
            return False
        if self.line is not None and h.line != self.line:
            return False
        return h.path == self.pattern or fnmatch.fnmatchcase(h.path, self.pattern)

    def label(self) -> str:
        loc = f"{self.pattern}:{self.line}" if self.line is not None else self.pattern
        return f"{loc} {self.rule}"


class AllowListError(ValueError):
    pass


def parse_allowlist(text: str) -> list[AllowEntry]:
    out: list[AllowEntry] = []
    for n, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        body, *rest = re.split(r"\s#", s, maxsplit=1)
        reason = rest[0].strip() if rest else ""
        parts = body.strip().rsplit(None, 1)
        if len(parts) != 2:
            raise AllowListError(f"allow-list line {n}: expected '<path>[:LINE] <RULE-ID> # reason'")
        target, rule = parts
        if rule not in RULES:
            raise AllowListError(f"allow-list line {n}: unknown rule id '{rule}'")
        if not reason:
            raise AllowListError(f"allow-list line {n}: a '# reason' is mandatory")
        m = re.match(r"^(.*?):(\d+)$", target)
        pattern, line = (m.group(1), int(m.group(2))) if m else (target, None)
        out.append(AllowEntry(pattern, line, rule, reason, n))
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
@dataclass
class Report:
    repo: str
    rev: str | None
    total_files: int = 0
    total_bytes: int = 0
    scanned_files: int = 0
    skipped_binary: int = 0
    skipped_never_open: int = 0
    skipped_large: int = 0
    missing: int = 0
    hits: list[Hit] = field(default_factory=list)
    allowed: list[Hit] = field(default_factory=list)
    unused_allow: list[str] = field(default_factory=list)
    extra: "Report | None" = None  # --extra-list paths, inspected as if tracked, reported apart
    extra_already_tracked: int = 0

    @property
    def passed(self) -> bool:
        return not self.blocking and (self.extra is None or not self.extra.blocking)

    @property
    def blocking(self) -> list[Hit]:
        return [h for h in self.hits if h.severity == BLOCK]

    @property
    def warnings(self) -> list[Hit]:
        return [h for h in self.hits if h.severity == WARN]

    def summary(self) -> list[dict]:
        rows: dict[str, dict] = {}
        for hs, key in ((self.hits, None), (self.allowed, "allowed")):
            for h in hs:
                r = rows.setdefault(h.rule, {"rule": h.rule, "block": 0, "warn": 0, "allowed": 0})
                r[key or h.severity] += h.count
        return sorted(rows.values(), key=lambda r: (-r["block"], -r["warn"], r["rule"]))

    def _body(self) -> dict:
        return {
            "totals": {
                "files": self.total_files, "bytes": self.total_bytes,
                "mib": round(self.total_bytes / MIB, 2), "content_scanned": self.scanned_files,
                "skipped_binary": self.skipped_binary,
                "skipped_never_open": self.skipped_never_open,
                "skipped_large": self.skipped_large, "missing": self.missing,
            },
            "summary": self.summary(),
            "blocking": [h.to_json() for h in self.blocking],
            "warnings": [h.to_json() for h in self.warnings],
            "allowed": [h.to_json() for h in self.allowed],
        }

    def to_json(self) -> dict:
        d = {"repo": self.repo, "rev": self.rev, "mode": "rev" if self.rev else "worktree",
             "passed": self.passed, "passed_tracked": not self.blocking, **self._body(),
             "unused_allow_entries": self.unused_allow}
        if self.extra is not None:
            d["extra"] = {"passed": not self.extra.blocking, **self.extra._body(),
                          "already_tracked_skipped": self.extra_already_tracked}
        return d


def extra_entries(repo: Path, paths: list[str], tracked: set[str]) -> tuple[list[Entry], int]:
    """Entries for --extra-list paths (files; directories are expanded), minus tracked ones."""
    out: dict[str, Entry] = {}
    skipped = 0

    def add(rel: str) -> None:
        nonlocal skipped
        if rel in tracked:
            skipped += 1
            return
        full = repo / rel
        try:
            st = os.lstat(_fs(full))
        except OSError:
            out[rel] = Entry(rel, "100644", "", None)
            return
        mode = "120000" if os.path.islink(_fs(full)) else "100644"
        out[rel] = Entry(rel, mode, "", st.st_size)

    for raw in paths:
        rel = raw.strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        if not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel) or ".." in rel.split("/"):
            raise ValueError(f"extra-list path must be repo-relative: {raw.strip()[:80]!r}")
        full = repo / rel
        if full.is_dir() and not os.path.islink(_fs(full)):
            for root, dirs, files in os.walk(_fs(full)):
                dirs[:] = [d for d in dirs if d != ".git"]
                for f in files:
                    sub = os.path.relpath(os.path.join(root, f), _fs(repo)).replace("\\", "/")
                    add(sub)
        else:
            add(rel)
    return list(out.values()), skipped


def _inspect(rep: Report, entries: list[Entry], reader, allow: list[AllowEntry],
             limits: Limits, want_context: bool, what: str) -> None:
    """Fill ``rep`` with the hits of ``entries`` (allow-list applied, counters shared)."""
    raw: list[Hit] = []
    for e in sorted(entries, key=lambda x: x.path):
        rep.total_files += 1
        if e.size is None:
            rep.missing += 1
            raw.append(_hit(e.path, "INFO-MISSING"))
            continue
        rep.total_bytes += e.size
        if e.mode == "160000":
            raw.append(_hit(e.path, "INFO-GITLINK"))
            continue
        if e.mode == "120000":
            raw.append(_hit(e.path, "INFO-SYMLINK"))
            continue
        if e.size > limits.hard:
            raw.append(_hit(e.path, "SIZE-HARD", f"{e.size / MIB:.1f} MiB"))
        elif e.size > limits.block:
            raw.append(_hit(e.path, "SIZE-BLOCK", f"{e.size / MIB:.1f} MiB"))
        elif e.size > limits.warn:
            raw.append(_hit(e.path, "SIZE-WARN", f"{e.size / MIB:.1f} MiB"))
        raw.extend(path_hits(e, limits))

        if is_never_open(e.path):
            rep.skipped_never_open += 1
            continue
        kind, data = reader.read(e, limits.block)
        if kind == "binary":
            rep.skipped_binary += 1
        elif kind == "large":
            rep.skipped_large += 1  # already a SIZE-BLOCK / SIZE-HARD hit
        elif kind == "missing":
            rep.missing += 1
            raw.append(_hit(e.path, "INFO-MISSING"))
        else:
            rep.scanned_files += 1
            raw.extend(scan_content(e.path, data, want_context))
    if rep.total_bytes > limits.total_warn:
        raw.append(_hit("(repository)", "SIZE-TOTAL",
                        f"{rep.total_bytes / MIB:.0f} MiB {what} in total"))
    for h in raw:
        ent = next((a for a in allow if a.matches(h)), None)
        if ent is None:
            rep.hits.append(h)
        else:
            ent.used += 1
            rep.allowed.append(h)


def run_checks(repo: Path, rev: str | None = None, allow_text: str | None = None,
               limits: Limits = Limits(), want_context: bool = False,
               extra_paths: list[str] | None = None) -> Report:
    """Inspect the tracked files (or ``rev``); ``extra_paths`` are inspected as if tracked
    but reported separately in ``Report.extra`` (working-tree mode only)."""
    repo = Path(repo)
    if extra_paths is not None and rev:
        raise ValueError("--extra-list needs the working tree and cannot be combined with --rev")
    entries = list_entries(repo, rev)
    reader = _RevReader(repo, rev) if rev else _WorktreeReader(repo)
    label = str(repo).replace("\\", "/")
    rep = Report(repo=label, rev=rev)
    try:
        if allow_text is None:
            blob = reader.read_path(ALLOW_FILE_REL)
            allow_text = blob.decode("utf-8", "replace") if blob is not None else ""
        allow = parse_allowlist(allow_text)
        _inspect(rep, entries, reader, allow, limits, want_context, "tracked")
        if extra_paths is not None:
            extras, rep.extra_already_tracked = extra_entries(
                repo, extra_paths, {e.path for e in entries})
            rep.extra = Report(repo=label, rev=None)
            _inspect(rep.extra, extras, reader, allow, limits, want_context, "listed")
    finally:
        reader.close()
    rep.unused_allow = [a.label() for a in allow if not a.used]
    return rep


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def _render_section(rep: Report, what: str, max_list: int) -> list[str]:
    t = rep._body()["totals"]
    lines = [
        f"{what} files: {t['files']}   total size: {t['mib']:.1f} MiB ({rep.total_bytes} bytes)",
        f"content scanned: {t['content_scanned']}   skipped: binary {t['skipped_binary']}, "
        f"never-open {t['skipped_never_open']}, too large {t['skipped_large']}, "
        f"missing {t['missing']}",
        "",
        f"{'RULE':<26}{'BLOCK':>7}{'WARN':>7}{'ALLOWED':>9}",
    ]
    tb = tw = ta = 0
    for r in rep.summary():
        lines.append(f"{r['rule']:<26}{r['block']:>7}{r['warn']:>7}{r['allowed']:>9}")
        tb, tw, ta = tb + r["block"], tw + r["warn"], ta + r["allowed"]
    lines.append(f"{'TOTAL':<26}{tb:>7}{tw:>7}{ta:>9}")
    for title, hs in (("BLOCKING", rep.blocking), ("WARN", rep.warnings)):
        if not hs:
            continue
        lines += ["", f"{title} hits ({what}):"]
        by_rule: dict[str, list[Hit]] = {}
        for h in hs:
            by_rule.setdefault(h.rule, []).append(h)
        for rule, group in sorted(by_rule.items()):
            lines.append(f"  [{rule}] {sum(h.count for h in group)} hit(s) in {len(group)} entr"
                         f"{'y' if len(group) == 1 else 'ies'}")
            for h in group if max_list <= 0 else group[:max_list]:
                loc = f"{h.path}:{h.line}" if h.line is not None else h.path
                lines.append(f"    {loc}" + (f"   ({h.detail})" if h.detail else ""))
                if h.context:
                    lines.append(f"        context: {h.context}")
            if 0 < max_list < len(group):
                lines.append(f"    ... {len(group) - max_list} more (use --max-list 0 or --json)")
    return lines


def render(rep: Report, max_list: int) -> str:
    lines = [f"prepush_check: {rep.repo} ({'rev ' + rep.rev if rep.rev else 'working tree'})"]
    lines += _render_section(rep, "tracked", max_list)
    if rep.extra is not None:
        lines += ["", f"=== EXTRA-LIST paths treated as tracked ({rep.extra_already_tracked} "
                      "already tracked, skipped) ==="]
        lines += _render_section(rep.extra, "extra", max_list)
    if rep.unused_allow:
        lines += ["", "allow-list entries that matched nothing (stale?):"]
        lines += [f"    {x}" for x in rep.unused_allow]
    parts = [("tracked", rep)] + ([("extra", rep.extra)] if rep.extra is not None else [])
    detail = ", ".join(f"{n}: {sum(h.count for h in r.blocking)} blocking" for n, r in parts)
    lines += ["", "RESULT: " + (f"FAIL ({detail})" if not rep.passed
                                else f"PASS (no blocking hits; {detail})")]
    return "\n".join(lines)


def _read_extra_list(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", help="repository root (default: the repo containing the cwd)")
    ap.add_argument("--rev", help="inspect this commit instead of the working tree")
    ap.add_argument("--allow", help=f"allow-list file (default: {ALLOW_FILE_REL} of the repo/rev)")
    ap.add_argument("--extra-list", metavar="FILE",
                    help="file of repo-relative paths (one per line, # comments) to inspect as if "
                         "tracked, reported separately; working tree only")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--context", action="store_true", help="show masked context per content hit")
    ap.add_argument("--max-list", type=int, default=40,
                    help="hits listed per rule in text output (0 = all)")
    ap.add_argument("--list-rules", action="store_true")
    args = ap.parse_args(argv)

    if args.list_rules:
        for rid, (sev, desc) in sorted(RULES.items()):
            print(f"{rid:<26}{sev:<7}{desc}")
        return 0
    try:
        if args.repo:
            repo = Path(args.repo)
        else:
            repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
        allow_text = None
        if args.allow:
            allow_text = Path(args.allow).read_text(encoding="utf-8")
        extra = _read_extra_list(args.extra_list) if args.extra_list else None
        rep = run_checks(repo, args.rev, allow_text, want_context=args.context, extra_paths=extra)
    except (GitError, AllowListError, OSError, ValueError) as exc:
        print(f"prepush_check: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(rep.to_json(), indent=1) if args.json else render(rep, args.max_list))
    return 0 if rep.passed else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    sys.exit(main())
