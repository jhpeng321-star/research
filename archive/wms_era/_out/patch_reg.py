p = 'E:/rd-wt/ws-r3-evalstates3/output/wd01/wm_n1/lpwo/make_registration_h60_real.py'
s = open(p).read()
s = s.replace('''    ap.add_argument("--code-dir", required=True)
''', '''    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--overlap-json", default=None)
''')
s = s.replace('''    (out / "REGISTRATION.json")''', '''    if args.overlap_json:
        reg["v2_overlap"] = json.loads(Path(args.overlap_json).read_text(encoding="utf-8"))
    (out / "REGISTRATION.json")''')
open(p, 'w').write(s)
