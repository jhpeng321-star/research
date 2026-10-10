"""Print the number of queue units that have no passed receipt yet."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upload_driver as u  # noqa: E402

def main():
    q = json.load(open(sys.argv[1], encoding="utf-8"))
    units = u.expand_queue(q["items"], q.get("exclude", []))
    done = u.load_receipts()
    pend = [x for x in units if u.safe_id(x["id"]) not in done]
    print(len(pend))

if __name__ == "__main__":
    main()
