import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
import scoped_archive1731 as m


class ScopedTests(unittest.TestCase):
    def test_only_exact_authorised_opaque_files_no_directory_or_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(m.base,'HOME',root):
                good=root/m.NAMES[0];good.write_bytes(b'opaque')
                self.assertTrue(m.safe(good))
                for name in ('e3final','WARM_BINDING.json','other.tar.gz'):
                    p=root/name;p.write_bytes(b'keep');self.assertFalse(m.safe(p))
                alias=root/m.NAMES[1];alias.symlink_to(good)
                self.assertFalse(m.safe(alias))

    def test_live_descendant_or_environment_reference_blocks(self):
        p=Path('/home/ubuntu')/m.NAMES[0]
        self.assertTrue(m.referenced(p,'CACHE='+str(p)))
        self.assertFalse(m.referenced(p,'/unrelated'))

    def test_existing_rolling_sha_removal_retains_source_on_bad_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/m.NAMES[0];p.write_bytes(b'opaque')
            e=dict(source=str(p),sha256='expected')
            with self.assertRaises(ValueError):m.rolling.remove_one(e,'different',{str(p)},set())
            self.assertTrue(p.exists())


if __name__=='__main__':unittest.main()
