import json
import argparse
import os
import random
from collections import defaultdict

CHUNKS_FILE = os.path.join("data", "chunks.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default=None, help="Filter to one section, e.g. safety")
    parser.add_argument("--brand", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--per-section", type=int, default=2, help="How many chunks to show per section")
    args = parser.parse_args()

    if not os.path.exists(CHUNKS_FILE):
        print("data/chunks.json not found. Run ingest.py first.")
        return

    with open(CHUNKS_FILE) as fh:
        chunks = json.load(fh)

    if args.section:
        chunks = [c for c in chunks if c["metadata"]["section"] == args.section]
    if args.brand:
        chunks = [c for c in chunks if c["metadata"]["brand"].lower() == args.brand.lower()]
    if args.model:
        chunks = [c for c in chunks if c["metadata"]["model"].lower() == args.model.lower()]

    by_section = defaultdict(list)
    for c in chunks:
        by_section[c["metadata"]["section"]].append(c)

    for section, group in sorted(by_section.items()):
        print(f"\n{'='*70}\nSECTION: {section}  ({len(group)} chunks)\n{'='*70}")
        sample = random.sample(group, min(args.per_section, len(group)))
        for c in sample:
            m = c["metadata"]
            print(f"\n[{m['brand']} {m['model']} | p.{m['page_number']} | {m['source_file']}]")
            print(c["text"][:400])
            print("..." if len(c["text"]) > 400 else "")

    print(f"\n\nTotal chunks matching filters: {len(chunks)}")
    print("Sections available:", sorted(by_section.keys()))


if __name__ == "__main__":
    main()
