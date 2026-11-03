"""Compile the actual remote launch/drain payloads without process control."""
import ast
import base64
import re
import shlex
import unittest
from unittest.mock import patch
from . import rolling_gpua_ops as m


class Intercepted(Exception):
    pass


class GPUAOperations(unittest.TestCase):
    def test_handover_waits_on_active_transfer_before_starting_bridge(self):
        events=[]
        with patch.object(m.ops,'EVIDENCE',__import__('pathlib').Path('/nonexistent-test-control')),patch.object(m,'drain',side_effect=[RuntimeError('prior archive transfer still active'),Intercepted]),patch.object(m.time,'sleep',side_effect=lambda _:events.append('wait')),patch.object(m,'bridge') as bridge:
            with self.assertRaises(Intercepted):m.handover()
            bridge.assert_not_called()
        self.assertEqual(events,['wait'])

    def test_actual_drain_payload_compiles_before_any_signal(self):
        def intercept(host, code, **kwargs):
            self.assertEqual(host,'cpuQ')
            tree=ast.parse(code)
            inner=next(ast.literal_eval(n.value) for n in tree.body
                       if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
            ast.parse(inner)
            self.assertIn('signal.SIGCONT);raise',inner)
            self.assertIn('prior archive transfer still active',inner)
            self.assertIn('assert completed or pre_proof',inner)
            self.assertIn('REMOTE_VERIFIED.json',inner)
            self.assertIn('prior mover finishing SHA-certified removals',inner)
            raise Intercepted
        with patch.object(m,'read',return_value={'pass_':True}),patch.object(m,'remote',side_effect=intercept):
            with self.assertRaises(Intercepted):m.drain()

    def test_actual_bridge_has_a_pinned_control_and_compiles(self):
        def intercept(command, **kwargs):
            payload=re.search(r"base64\.b64decode\('([A-Za-z0-9+/=]+)'\)",shlex.split(command[-1])[-1]).group(1)
            code=base64.b64decode(payload).decode()
            ast.parse(code)
            self.assertIn('COOPMWM_ARCHIVE_CONTROL=',code)
            self.assertIn('PRIOR_DRAINED.json',code)
            self.assertIn("'AUTHORITY.json'",code)
            raise Intercepted
        with patch('subprocess.run',side_effect=intercept):
            with self.assertRaises(Intercepted):m.bridge()

    def test_actual_route_is_metadata_only_and_uses_gpua_floor(self):
        def intercept(host,code,**kwargs):
            ast.parse(code)
            self.assertEqual(host,'cpuQ')
            self.assertIn(str(8*1024**3),code)
            self.assertIn('connect.bjb1.seetacloud.com',code)
            self.assertNotIn('os.kill',code)
            self.assertNotIn('unlink',code)
            raise Intercepted
        with patch.object(m,'remote',side_effect=intercept):
            with self.assertRaises(Intercepted):m.route()


if __name__=='__main__':unittest.main()
