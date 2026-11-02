from . import niche_ops


def test_qualification_payload_compiles_without_remote_execution(monkeypatch, capsys):
    def validate(host, code, **kwargs):
        assert host == 'cpuQ'
        compile(code, '<joint-niche-nullfix-qualification>', 'exec')
        assert 'FROZEN_GEOMETRY_SOURCE.py' in code
        assert 'test_nullfix_cpu' in code
        return 'syntax_valid_no_dispatch'
    monkeypatch.setattr(niche_ops, 'remote', validate)
    niche_ops.qualify()
    assert capsys.readouterr().out.strip() == 'syntax_valid_no_dispatch'
