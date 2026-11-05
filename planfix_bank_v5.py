"""Six untouched inputs only; preserve both charged startup failures."""
from planfix_bank_v4 import inactive_training,pace_while_owner_stopped
import planfix_run_bank as bank

original_verify=bank.verify_plan


def remaining_only(item,cfg,native,registration,workers):
    failure=bank.lt.read(bank.ROOT/'plan_fix/ENVIRONMENT_FAILED_INPUT_V4.json')
    if failure['input_sha256']==bank.digest(item):
        return dict(input_sha256=bank.digest(item),move=item['move'],charged_sim_s=2400,
            rows=[dict(source_example_id=item['example_id'],original_gain_veh_s=item['original_gain_veh_s'],
                translated_gain_veh_s=None,family=item['family'],world_seed=item['world_seed'],
                seed_index=item['source_seed_index'],admitted=False,cooperative=False,keep=False,
                status='verification_failed',reason='v4 missing staged adapter config at startup; no replay, charges retained')])
    return original_verify(item,cfg,native,registration,workers)


if __name__=='__main__':
    bank.ARTIFACTS='bank_verification_v5'
    bank.check_pace=pace_while_owner_stopped
    bank.verify_plan=remaining_only
    bank.main()
