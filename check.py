"""End-to-end check of the StudyMate AI Gemma 4 pipeline (no UI).

Usage:
    python check.py                              # explain mode, typed topic
    python check.py --mode quiz                  # quiz mode
    python check.py --mode summarise             # summarise pasted notes
    python check.py --mode doubt                 # walk through a doubt
    python check.py --mode plan                  # day-by-day study plan
    python check.py --image samples/sample-notes.png        # explain a photo
    python check.py --mode quiz --image samples/sample-notes.png
    python check.py --all                        # one live call per mode

Every mode sends a real request to Gemma 4 through the Gemini API.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gemma_client as gc

SAMPLE = "samples/sample-notes.png"

DEFAULTS = {
    "explain": dict(topic="Binary Search"),
    "quiz": dict(topic="Binary Search", num_mcqs=3),
    "summarise": dict(
        notes=(
            "A binary search tree is a binary tree where every node's left subtree holds smaller "
            "keys and its right subtree holds larger keys. Search, insert and delete all follow "
            "one root-to-leaf path, so they run in O(log n) when the tree is balanced. An in-order "
            "traversal of a BST returns the keys in sorted order."
        )
    ),
    "doubt": dict(question="I don't understand recursion. Walk me through it."),
    "plan": dict(plan_topics="DBMS\nOperating Systems\nComputer Networks\nData Structures", days=5),
}


def summarize(pack: gc.StudyPack) -> None:
    print(f"\nMode:  {pack.mode}")
    print(f"Topic: {pack.topic}")
    if pack.detected_text:
        print(f"Read from image: {pack.detected_text}")
    for line in (pack.explanation or pack.summary).splitlines()[:3]:
        if line.strip():
            print(f"  {line.strip()}")
    if pack.analogy:
        print(f"Analogy: {pack.analogy[:150]}")
    if pack.example:
        print(f"Example: {pack.example[:150]}")
    if pack.code:
        print(f"Code:    {len(pack.code.splitlines())} line(s)")
    print(
        f"Key points: {len(pack.key_points)} | Keywords: {len(pack.keywords)} | "
        f"MCQs: {len(pack.mcqs)} | Dry-run steps: {len(pack.dry_run)} | Plan days: {len(pack.plan)}"
    )
    for i, q in enumerate(pack.mcqs, 1):
        print(f"  Q{i}. {q.question} -> {q.options[q.answer_index]}")
    for d in pack.plan:
        print(f"  {d.day}: {d.focus} ({len(d.tasks)} tasks)")


def run(mode: str, image: Path | None) -> gc.StudyPack:
    kwargs = dict(DEFAULTS[mode])
    if image:
        kwargs.pop("topic", None)
        kwargs["image_bytes"] = image.read_bytes()
        kwargs["image_mime_type"] = "image/png"
    return gc.generate_study_pack(mode=mode, difficulty="Intermediate", language="English", **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the StudyMate AI Gemma 4 pipeline.")
    parser.add_argument("--mode", choices=gc.MODES, default="explain")
    parser.add_argument("--image", type=Path, help=f"image to read (e.g. {SAMPLE})")
    parser.add_argument("--all", action="store_true", help="run one live call per mode")
    args = parser.parse_args()

    print(f"Model: {gc.get_model()}")
    key = gc.get_api_key() or ""
    print(f"Key:   {key[:5]}...{key[-4:]} (masked)")

    modes = gc.MODES if args.all else [args.mode]
    failures = 0
    for mode in modes:
        image = args.image or (Path(SAMPLE) if args.all and mode == "explain" else None)
        try:
            summarize(run(mode, image))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"\nMode:  {mode}\nFAILED: {exc}")

    print("\nCHECK OK" if not failures else f"\nCHECK FAILED ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
