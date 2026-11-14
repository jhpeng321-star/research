"""MAIN13:35 additive B-native/public delivery; preserve completed V1 analysis."""
import argparse
from . import interim_u8_ops as parent


def setup():
    parent.SOURCE = '/home/ubuntu/coopmwm20261103_interimanalysis2'
    parent.CONTROL = parent.ROOT+'/interim_analysis_1335_v2'
    parent.REG = 'INTERIM_U8_ANALYSIS_REGISTRATION_1335_V2.json'
    parent.CPU = 'INTERIM_U8_ANALYSIS_CPU_1335_V2.json'
    parent.DISPATCH = 'INTERIM_U8_ANALYSIS_DISPATCH_1335_V2.json'
    parent.DELIVERED = 'INTERIM_U8_ANALYSIS_DELIVERED_1335_V2.json'
    parent.AUTHORITY = 'MAIN2026-11-03 13:35'
    parent.MAIN_TARGET = 'E:/agent-work/_sol/COOPMWM_INTERIM_U8_MAIN_V2.json'
    parent.PUBLISH_PUBLIC = True


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=('qualify','register','launch','status','deliver'))
    setup();getattr(parent, p.parse_args().action)()
