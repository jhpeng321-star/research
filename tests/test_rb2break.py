"""Development guard must run before opening a reserved job, not after loading it."""
import unittest

from tools.rb2break.run import DEV, filtered_starts


class DevelopmentGuardTest(unittest.TestCase):
    def test_no_reserved_job_is_opened(self):
        code = '''def extract(worlds, opened):
    for i, world in enumerate(worlds):
        opened.append(world['seed'])
'''
        namespace = {}
        exec(compile(filtered_starts(code, [969420, 969431]), '<test>', 'exec'), namespace)
        opened = []
        namespace['extract']([{'seed': s} for s in range(969400, 969440)], opened)
        self.assertEqual(opened, [969420, 969431])

    def test_reject_reserved_selection(self):
        with self.assertRaises(ValueError):
            filtered_starts('for i, world in enumerate(worlds): pass', [969432])

    def test_selector_fails_closed_on_source_change(self):
        with self.assertRaises(ValueError):
            filtered_starts('for world in worlds: pass', DEV)


if __name__ == '__main__':
    unittest.main()
