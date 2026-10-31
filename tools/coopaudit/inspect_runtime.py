import inspect
import sys
from pathlib import Path
R = Path('/root/autodl-tmp/c4/speedhead20261030/e3final')
sys.path[:0] = ['/root/wmdata2', str(R/'runtime/src'), str(R/'runtime'), '/root/wmdata2/tools/speedhead']
from traffic_shepherd.c4 import branching, evalprep
from traffic_shepherd.c4.env import CavSumoEnv
from wms.control.nativeinit import NativeResidualCommand
from observer import Measurement
print('branching', branching.__file__)
print(inspect.getsource(evalprep.run_e3))
print(inspect.getsource(Measurement.finish))
print(inspect.getsource(CavSumoEnv._accel_by_vehicle))
print(inspect.getsource(NativeResidualCommand))
