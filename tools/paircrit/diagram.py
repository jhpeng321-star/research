"""Render the measured reliability bins as a small, dependency-free SVG artifact."""
import html
import json
from pathlib import Path

root = Path('/root/autodl-tmp/c4/paircrit20261026')
data = json.loads((root / 'RELIABILITY.json').read_text())
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="670" viewBox="0 0 960 670">',
    '<rect width="960" height="670" fill="white"/>',
    '<g font-family="sans-serif" fill="#222">',
    '<text x="30" y="28" font-size="20">S-PAIRCRIT: breakdown reliability after temperature calibration</text>',
    '<text x="30" y="52" font-size="13">TRAIN development worlds 968137/968138; only two initially unbroken outcomes per output.</text>',
    '<text x="30" y="74" font-size="13">Qualification FAILED. Zero Brier at120/300s is degenerate: every eligible outcome is an event.</text>']
for arm, arm_name in enumerate(('native', 'policy')):
    for column, horizon in enumerate((60, 120, 300)):
        d = data[f'{arm_name}_{horizon}']
        x, y, size = 55 + 310 * column, 125 + 265 * arm, 190
        parts += [f'<text x="{x}" y="{y-15}" font-size="15">{arm_name.upper()} within {horizon}s; Brier {d["brier"]:.3f}</text>',
            f'<path d="M{x},{y} V{y+size} H{x+size}" fill="none" stroke="#555"/>',
            f'<path d="M{x},{y+size} L{x+size},{y}" stroke="#888" stroke-dasharray="4 4"/>']
        for tick in (0, .5, 1):
            parts += [f'<text x="{x+size*tick-4}" y="{y+size+18}" font-size="11">{tick:g}</text>',
                f'<text x="{x-25}" y="{y+size-size*tick+4}" font-size="11">{tick:g}</text>']
        for row in d['bins']:
            if row['n']:
                px, py = x + size * row['predicted'], y + size * (1 - row['observed'])
                tip = html.escape(f"n={row['n']}; predicted={row['predicted']:.5f}; observed={row['observed']:.5f}")
                parts.append(f'<circle cx="{px}" cy="{py}" r="5" fill="#165c7d"><title>{tip}</title></circle>')
                parts.append(f'<text x="{px-24}" y="{py+20}" font-size="11">n={row["n"]}</text>')
        parts += [f'<text x="{x+5}" y="{y+size+37}" font-size="12">Mean predicted probability [0,1]</text>',
            f'<text transform="translate({x-35},{y+size-5}) rotate(-90)" font-size="12">Observed event fraction [0,1]</text>']
parts += ['<text x="30" y="652" font-size="12">Source: frozen cycle0 SUMO traces; rule registered2026-10-26. Points: ten fixed probability bins; dashed line: perfect calibration.</text>',
          '</g></svg>']
(root / 'reliability.svg').write_text('\n'.join(parts))
print('reliability.svg generated from RELIABILITY.json')
