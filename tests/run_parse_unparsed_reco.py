"""Parse the existing `recommendations/unparsed_reco.json` created by the chooser
and persist a cleaned, parsed JSON file using the project's utilities.

Run locally with:

    python tests/run_parse_unparsed_reco.py

This script is intentionally simple so you can run it without the Featherless
API or any LLM client installed.
"""

import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plot_type_generator.utils import extract_json_content, save_recommendations


def main():
    repo_root = project_root
    src_reco = (
        repo_root / "plot_type_generator" / "recommendations" / "unparsed_reco.json"
    )
    if not src_reco.exists():
        print("unparsed_reco.json not found at:", src_reco)
        raise SystemExit(1)

    raw = src_reco.read_text(encoding="utf-8")
    if not raw.strip():
        print(
            "Warning: `unparsed_reco.json` is empty. Falling back to embedded sample."
        )
        # Minimal fallback that mirrors the structure produced by the LLM provider
        raw = (
            "content='{'"
            '\n  "plot type": "line",'
            '\n  "features to be selected": ["date", "sales_amount"],'
            '\n  "suggestions": {"primary_features": ["date.month", "sales_amount"]}'
            "\n}' additional_kwargs={'refusal': None}"
        )
    print("Read unparsed file (first 200 chars):\n", raw[:200])

    try:
        parsed = extract_json_content(raw)
    except ValueError as e:
        print("Failed to parse JSON:", e)
        raise

    print("Parsed JSON keys:", list(parsed.keys()))

    out_dir = repo_root / "plot_type_generator" / "recommendations"
    saved = save_recommendations(parsed, out_dir=str(out_dir))
    print("Saved parsed recommendations to:", saved)


if __name__ == "__main__":
    main()
