"""Aggregate py-spy raw (collapsed) stacks: self time and inclusive time per function."""
import collections, sys, re
inc, self_ = collections.Counter(), collections.Counter()
total = 0
for line in open(sys.argv[1], encoding='utf-8', errors='replace'):
    line = line.rstrip('\n')
    if not line:
        continue
    stack, _, n = line.rpartition(' ')
    n = int(n); total += n
    frames = stack.split(';')
    names = [re.sub(r'\s*\(([^)]*?)(:\d+)?\)$', r' (\1)', f) for f in frames]
    for f in set(names):
        inc[f] += n
    self_[names[-1]] += n
key = sys.argv[2] if len(sys.argv) > 2 else None
print('total samples', total)
print('--- inclusive top 45')
for f, n in inc.most_common(45):
    print(f'{100*n/total:5.1f}%  {f}')
print('--- self top 30')
for f, n in self_.most_common(30):
    print(f'{100*n/total:5.1f}%  {f}')
