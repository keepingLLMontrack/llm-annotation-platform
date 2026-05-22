from __future__ import annotations

import argparse

from datasets import load_dataset

from app import DEFAULT_ANNOTATION_REPO, DEFAULT_SOURCE_DATASET, DEFAULT_SOURCE_SPLIT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--annotation-repo", default=DEFAULT_ANNOTATION_REPO)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records = load_dataset(args.source, split=args.split)
    if args.limit:
        records = records.select(range(min(len(records), args.limit)))

    print(f"Loaded {len(records)} source records from {args.source}/{args.split}")
    print(f"Annotation repo: {args.annotation_repo}")
    print("Open the Streamlit app and submit annotations there.")


if __name__ == "__main__":
    main()
