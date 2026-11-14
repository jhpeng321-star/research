"""Transport payload must compile at both Python layers before any restore."""
import ast
import contextlib
import io
import json
import unittest
from unittest.mock import patch
from . import bank_restore_ops as m


class RestorePayload(unittest.TestCase):
    def test_nested_stdin_worker_compiles_without_any_remote_execution(self):
        seen=[]
        def remote(host,code,timeout):
            tree=ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node,ast.keyword) and node.arg=='input':
                    worker=ast.literal_eval(node.value);compile(worker,'<restore-worker>','exec');seen.append(worker)
            return json.dumps(dict(new_sim_s=0,syntax_only=True))
        with patch.object(m,'remote',remote),patch.object(m,'write'),contextlib.redirect_stdout(io.StringIO()):
            m.restore()
        self.assertEqual(len(seen),1)
        self.assertIn("m.base.digest_file(temp)==e['sha256']",seen[0])


if __name__=='__main__':unittest.main()
