"""Publication safeguards: missing coverage is explicit, no partial-run inference."""
import unittest
from unittest.mock import patch
from tools.coopaudit import finalize_v2 as F


class PublicationContracts(unittest.TestCase):
    def test_incomplete_remote_receipt_never_collects_or_commits(self):
        from types import SimpleNamespace
        with patch.object(F,'git',return_value='frozen'),patch.object(F,'run',return_value=SimpleNamespace(returncode=1,stderr='incomplete')),patch.object(F,'commit') as commit:
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                F.finish('frozen')
            commit.assert_not_called()

    def test_complete_template_discloses_conditional_ci_and_no_zero_imputation(self):
        rows=[]
        for cohort in ('eligible21','eligible_high_share20'):
            for outcome in ('300','600','local'):
                for c in ('mappo','wall','sumocf'):
                    for metric in ('C_G','D_GROUP','sum_D_ONLY','frac_positive'):
                        rows.append(dict(cohort=cohort,outcome=outcome,controller=c,metric=metric,mean=0.,lo=-1.,hi=1.))
        s=dict(summary=rows,paired=[dict(metric='C_G',outcome=o,mean=0.,lo=-1.,hi=1.) for o in ('300','600','local')],
               safety={c:dict(ttc_below_3_vs=0,hard_braking_vs=0) for c in ('native','mappo','wall','sumocf')},
               actual_total_sim_s=243480,charged_sim_s=243780)
        text=F.report(s)
        self.assertLessEqual(len(text.splitlines()),60)
        self.assertIn('UNDEFINED (missing strata)',text)
        self.assertIn('conditional eligible21-world',text)
        self.assertIn('never treat missing cooperation as zero',text)
        self.assertIn('One mechanics world, NOT panel inference',text)


if __name__=='__main__':
    unittest.main(verbosity=2)
