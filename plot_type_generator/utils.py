import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _load_prompt(prompt_name: str) -> str:
    """Load the query refiner prompt file from the prompts directory.

    Returns the raw prompt text. Raises FileNotFoundError if file missing.
    """
    base = os.path.dirname(__file__)
    prom_path = os.path.join(base, "prompts", prompt_name)
    with open(prom_path, "r", encoding="utf-8") as f:
        return f.read()


def _get_api_key() -> str:
    """Read Featherless API key from env.

    Raises RuntimeError if not set to avoid leaking keys in source.
    """
    key = os.environ.get("FEATHERLESS_API_KEY")
    if not key:
        raise RuntimeError(
            "FEATHERLESS_API_KEY environment variable is not set."
            " Set it before running this agent."
        )
    return key


def extract_json_content(response_string):
    """
    Extract the JSON content from a response string.

    Args:
        response_string: String containing JSON data, possibly wrapped in markdown code blocks

    Returns:
        dict: Parsed JSON object
    """
    # Strip whitespace
    content = response_string.strip()

    # Remove **{{{ ... }}** markers
    if content.startswith("**{{{") and content.endswith("}}**"):
        content = content[5:-4].strip()
    # Remove ```json and ``` markers
    elif content.startswith("```"):
        # Find the first newline after opening ```
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        # Remove trailing ```
        if content.endswith("```"):
            content = content[:-3].rstrip()

    # Try to parse the JSON directly
    try:
        json_data = json.loads(content)
        return json_data
    except json.JSONDecodeError as e:
        # Try to extract JSON by finding balanced braces
        start_idx = content.find('{')
        if start_idx != -1:
            # Find the matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(content)):
                char = content[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the matching brace
                            json_str = content[start_idx:i+1]
                            try:
                                json_data = json.loads(json_str)
                                return json_data
                            except json.JSONDecodeError:
                                break

        # If that fails, try the old method for backward compatibility
        # Use regex to find content between content=' and the next ' that ends the field
        match = re.search(
            r"content='(.*?)'\s+additional_kwargs=", response_string, re.DOTALL
        )

        if not match:
            raise ValueError(f"Failed to parse JSON content: {e}")

        # Get the escaped JSON string
        escaped_json = match.group(1)

        # Unescape the JSON string (replace \n with actual newlines, etc.)
        unescaped_json = escaped_json.encode().decode("unicode_escape")

        # Parse the JSON
        try:
            json_data = json.loads(unescaped_json)
            return json_data
        except json.JSONDecodeError as e2:
            raise ValueError(f"Failed to parse JSON content: {e2}")


def save_recommendations(
    parsed: Dict[str, Any], out_dir: str = "./recommendations"
) -> str:
    """Persist parsed recommendations to a timestamped JSON file.

    Returns the path to the written file.
    """
    base = Path(__file__).resolve().parent
    rec_dir = Path(out_dir) if out_dir else base / "recommendations"
    rec_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fn = rec_dir / f"plot_recommendations_{ts}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    return str(fn)
