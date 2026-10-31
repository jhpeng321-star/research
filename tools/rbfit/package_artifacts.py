"""Lossless publication of a large finite-support posterior JSON."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    root=args.directory
    source=root/"idm_posteriors.json"
    raw=source.read_bytes()
    parsed=json.loads(raw)
    if parsed["schema"]!="wm_phys.idm_posteriors.v1":
        raise ValueError("Unexpected posterior schema")
    compressed=gzip.compress(raw,compresslevel=9,mtime=0)
    assert gzip.decompress(compressed)==raw
    out=root/"idm_posteriors.json.gz"
    with out.open("xb") as stream:
        stream.write(compressed)
    receipt={"format":"gzip-compressed UTF-8 JSON, lossless original bytes",
             "schema":parsed["schema"],"n_drivers":len(parsed["drivers"]),
             "raw_sha256":hashlib.sha256(raw).hexdigest(),"raw_bytes":len(raw),
             "compressed_sha256":hashlib.sha256(compressed).hexdigest(),"compressed_bytes":len(compressed),
             "original_preserved":True}
    (root/"posterior_publication.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    main()
