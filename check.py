"""Quick end-to-end check of the StudyMate AI Gemma 4 pipeline (no UI).

Usage:
    python check.py                 # topic mode
    python check.py samples/sample-notes.png   # image mode (multimodal)
"""
from __future__ import annotations

import sys
from pathlib import Path

import gemma_client as gc


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    print(f"Model: {gc.get_model()}")
    print(f"Key:   {(gc.get_api_key() or '')[:5]}...{(gc.get_api_key() or '')[-4:]} (masked)")

    if image_path:
        print(f"Mode:  IMAGE ({image_path})")
        pack = gc.generate_study_pack(
            image_bytes=image_path.read_bytes(),
            image_mime_type="image/png",
            difficulty="Beginner",
            language="Hinglish",
            num_mcqs=3,
        )
    else:
        print("Mode:  TOPIC")
        pack = gc.generate_study_pack(
            topic="Binary Search",
            difficulty="Exam-ready",
            language="English",
            num_mcqs=3,
        )

    print(f"\nTopic: {pack.topic}")
    if pack.detected_text:
        print(f"Read from image: {pack.detected_text}")
    print(f"Explanation: {pack.explanation[:220]}...")
    print(f"Analogy: {pack.analogy[:160]}")
    print(f"Key points: {len(pack.key_points)} | Keywords: {len(pack.keywords)}")
    print(f"MCQs: {len(pack.mcqs)}")
    for i, q in enumerate(pack.mcqs, 1):
        print(f"  Q{i}. {q.question}")
        print(f"      -> {q.options[q.answer_index]}")
    print("\nCHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
