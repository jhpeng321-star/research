"""CV17 predecessor contract rejection without changing preserved evidence."""
import json
from pathlib import Path
import pytest
from output.controller_verification_v17 import prepare_registration as prep

@pytest.mark.parametrize("field,value", [("stopped",False),("amount_s",0),("completed_simulated_s",190.0),("status","complete")])
def test_predecessor_mutation_is_rejected(monkeypatch,field,value):
    original=Path.read_text
    target=(prep.ROOT/prep.PRIOR_LEDGER).resolve()
    def changed(path,*args,**kwargs):
        text=original(path,*args,**kwargs)
        if path.resolve()==target:
            data=json.loads(text)
            if field=="stopped":
                data["campaigns"][prep.PRIOR_CAMPAIGN][field]=value
            else:
                data["reservations"][0][field]=value
            return json.dumps(data)
        return text
    monkeypatch.setattr(Path,"read_text",changed)
    with pytest.raises(ValueError,match="CV15"):
        prep.build_registration()
