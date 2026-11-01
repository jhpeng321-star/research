import json

import pytest

from tools.rb2int.report import render,interval


def test_a_partial_comparison_cannot_be_published_as_complete_final_delivery(tmp_path):
    path=tmp_path/'comparison.json'
    path.write_text(json.dumps(dict(complete_registered_scope=False,scope='primary54',matched_complete_worlds=[])))
    with pytest.raises(ValueError,match='complete registered primary'):
        render(tmp_path,path)
    path.write_text(json.dumps(dict(complete_registered_scope=True,scope='primary54',matched_complete_worlds=[969550])))
    with pytest.raises(ValueError,match='complete registered primary'):
        render(tmp_path,path)


def test_unavailable_rate_ci_remains_na_and_probability_units_convert_to_pp():
    assert interval(dict(mean=.05625,low=None,high=None),100,2)=='5.62 [NA, NA]'
    assert interval(dict(mean=None,low=None,high=None))=='NA [NA, NA]'
