from .ls_label_ops import variation_summary


def test_partial_and_skipped_slates_do_not_dilute_null_rate():
    rows=[dict(status='rejected',candidate_count=8,exact_null=True),
          dict(status='accepted',candidate_count=8,exact_null=False),
          dict(status='rejected',candidate_count=3,exact_null=None),
          dict(status='inactionable',candidate_count=0,exact_null=None)]
    out=variation_summary(rows)
    assert out['before_release']['exact_null_rate']==.5
    assert out['before_release']['partial_or_unmeasured_lists']==2
    assert out['before_release']['inactionable_skipped']==1
    assert out['niches1']['exact_null_rate'] is None
    assert out['niches4']['mean_pairwise_distance'] is None


def test_release_cohorts_require_observed_clamp_and_keep_niches_separate():
    base=dict(status='accepted',candidate_count=8,exact_null=False)
    rows=[dict(base,niches=1,search_clamp_enabled=False),
          dict(base,search_clamp_enabled=True),
          dict(base,niches=1,search_clamp_enabled=True,mean_pairwise_distance=2.),
          dict(base,niches=4,search_clamp_enabled=True,mean_pairwise_distance=4.,occupied_niches=4)]
    out=variation_summary(rows)
    assert [out[k]['labelled_lists'] for k in ('before_release','niches1','niches4')]==[1,1,1]
    assert out['niches1']['mean_pairwise_distance']==2.
    assert out['niches4']['mean_occupied_niches']==4
    assert out['clamp_without_niche_config']['labelled_lists']==1
