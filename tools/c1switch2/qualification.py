"""MAIN23:28 WM-PHYS eligibility rule; never reads switching outcomes."""


def qualifies(receipt):
    if receipt.get('panel') != 'S-RB-EVAL' or not receipt.get('complete'):
        raise ValueError('complete registered S-RB-EVAL receipt required')
    wm, phys, real = (receipt[k] for k in ('wm2', 'wm_phys', 'real'))
    import math
    values = [model['delay_crps'][str(h)] for model in (wm, phys) for h in (120, 300, 600)]
    values += [wm['breakdown_rate'], phys['breakdown_rate'], real['breakdown_rate'],
               wm['hdv_position_error_60_m'], phys['hdv_position_error_60_m']]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('finite qualification metrics required')
    return (all(phys['delay_crps'][str(h)] < wm['delay_crps'][str(h)] for h in (120, 300, 600))
            and abs(phys['breakdown_rate']-real['breakdown_rate']) <
                abs(wm['breakdown_rate']-real['breakdown_rate'])
            and phys['hdv_position_error_60_m'] <= wm['hdv_position_error_60_m'])
