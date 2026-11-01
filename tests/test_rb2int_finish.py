import pytest

from tools.rb2int import finish


def test_delivery_cannot_stage_over_foreign_work(monkeypatch):
    calls=[]
    def command(argv,cwd=None):
        calls.append(argv)
        if argv==['git','branch','--show-current']: return 'c4/rb2-int\n'
        if argv==['git','diff','--cached','--name-only']: return 'user_file.py\n'
        return ''
    monkeypatch.setattr(finish,'command',command)
    with pytest.raises(ValueError,match='foreign staging'):
        finish.git_commit([finish.ROOT/'HANDOFF.md'],'test')
    assert not any(c[:2]==['git','add'] for c in calls)


def test_delivery_may_not_commit_source_code(monkeypatch):
    calls=[]
    def command(argv,cwd=None):
        calls.append(argv)
        if argv==['git','branch','--show-current']: return 'c4/rb2-int\n'
        return ''
    monkeypatch.setattr(finish,'command',command)
    with pytest.raises(ValueError,match='own named evidence'):
        finish.git_commit([finish.ROOT/'src/wms/model/wms.py'],'test')
    assert not any(c[:2]==['git','add'] for c in calls)
