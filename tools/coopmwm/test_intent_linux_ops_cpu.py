"""Release plumbing checks: no remote work on bad identity, exact frozen fixtures."""
import json
import subprocess
from unittest.mock import patch

import pytest

from . import intent_linux_ops as m
from . import v2v_qualification_ops as q


def test_bad_or_dirty_identity_never_dispatches():
    with patch.object(m, 'git') as git, patch.object(q, 'remote') as remote:
        for value in ('', 'cf3efc9e', 'x'*40):
            with pytest.raises(ValueError):
                m.release_options(value)
        git.return_value = b'b'*40+b'\n'
        with pytest.raises(ValueError, match='HEAD differs'):
            m.release_options('a'*40)
        git.side_effect = [b'a'*40+b'\n', subprocess.CalledProcessError(1, ['git', 'diff'])]
        with pytest.raises(subprocess.CalledProcessError):
            m.release_options('a'*40)
        remote.assert_not_called()


def test_frozen_identity_uses_exact_pre_intent_blobs():
    commit = 'a'*40
    seen = []

    def git(*args):
        seen.append(args)
        if args[:2] == ('rev-parse', 'HEAD'):
            return (commit+'\n').encode()
        if args[0] == 'rev-parse':
            return b'c'*40+b'\n'
        if args[0] == 'show':
            return ('# exact '+args[1]+'\n').encode()
        return b''

    with patch.object(m, 'git', git):
        value = m.release_options(commit)
    assert value['source_commit'] == commit
    assert len(value['frozen_files']) == 7
    for name in m.BASELINE_NAMES:
        assert value['frozen_files']['intent_baseline/'+name+'.py'] == (
            '# exact cf3efc9e:tools/coopmwm/'+name+'.py\n').encode()
    assert any(args[:2] == ('diff', '--exit-code') for args in seen)
    assert value['frozen_files']['coop_baseline/mppi_search.py'] == (
        '# exact b20bec2f:tools/coopmwm/mppi_search.py\n').encode()
    assert value['extra_env']['COOPMWM_COOP_BASELINE'].endswith('/coop_baseline/mppi_search.py')
    assert sum(args[0] == 'cat-file' for args in seen) == len(set(q.FILES+m.EXTRA_FILES))


def test_generation_cannot_reuse_legacy_root():
    with patch.object(q, 'BASE'), patch.object(q, 'SOURCE'), patch.object(q, 'GENERATION'):
        for name in ('V8', '', '../I1', 'I0', 'I1/something'):
            with pytest.raises(ValueError):
                m.configure(name)
        m.configure('I2')
        assert q.BASE == m.BASE_SOURCE
        assert q.SOURCE.endswith('intentcpu_i2') and q.GENERATION == 'INTENT_I2'


def test_qualifier_accepts_pinned_fixture_payload_without_spawning(tmp_path):
    fixture = b'# frozen baseline\n'
    captured = []

    def remote(host, code, timeout):
        assert host == 'cpuQ'
        compile(code, '<remote qualification>', 'exec')
        captured.append(code)
        return json.dumps(dict(pid=123, sim_s=0))

    with patch.object(q, 'FILES', ()), patch.object(q.ops, 'LOCAL', tmp_path), \
            patch.object(q.ops, 'EVIDENCE', tmp_path), patch.object(q, 'remote', remote), \
            patch.object(q, 'write') as write:
        q.qualify(frozen_files={'intent_baseline/base.py': fixture},
            additional_tests=['test_extra_cpu'], integration='intent release',
            source_commit='a'*40, extra_env={'COOPMWM_INTENT_BASELINE_DIR': '/safe/baseline'})
    assert len(captured) == 1 and write.call_count == 1
    assert 'intent_baseline/base.py' in captured[0]
    assert 'COOPMWM_INTENT_BASELINE_DIR' in captured[0]
    assert "tests+=['test_extra_cpu']" in captured[0]


def test_fixture_cannot_escape_release_root():
    with patch.object(q, 'FILES', ()), patch.object(q, 'remote') as remote:
        for name in ('../base.py', '/tmp/base.py', 'C:/base.py', 'dir\\base.py'):
            with pytest.raises(ValueError):
                q.qualify(frozen_files={name: b'# invalid'})
        remote.assert_not_called()
