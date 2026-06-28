#!/usr/bin/env python3
"""
Client-side ReSiReg inference script.

Reads a local image, sends it to the local OpenAI-compatible ReSiReg server,
and saves the returned result image.

Usage:
    python client_infer.py --image /path/to/image.jpg --mode traverse
    python client_infer.py --image /path/to/image.jpg --mode segment --classes "wooden bridge,tree,water"
    python client_infer.py --image /path/to/image.jpg --mode similarity --positive "wooden bridge" --negative "tree,water"
    python client_infer.py --image /path/to/image.jpg --mode language --positive "wooden bridge" --negative "tree,water"
"""

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path


def image_to_data_url(path: str) -> str:
    """Encode an image file as a base64 data URL, or pass through an http(s) URL."""
    if path.startswith(("http://", "https://")):
        return path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = p.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")

    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_prompt(args: argparse.Namespace) -> str:
    """Construct the prompt text for the selected mode."""
    mode = args.mode.lower()
    if mode == "segment":
        return f"/segment: {args.classes}"
    if mode == "traverse":
        return f"/traverse: {args.positive} vs {args.negative}"
    if mode == "similarity":
        return f"/similarity: {args.positive} - {args.negative}"
    if mode == "language":
        return f"{args.positive} vs {args.negative}"
    raise ValueError(f"Unknown mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReSiReg inference modes via the local server."
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path or URL to the input image.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["traverse", "segment", "similarity", "language"],
        default="traverse",
        help="Inference mode to run (default: traverse).",
    )
    parser.add_argument(
        "--positive", "-p",
        default="wooden bridge",
        help="Positive prompt (used by traverse, similarity, language).",
    )
    parser.add_argument(
        "--negative", "-n",
        default="tree,water",
        help="Negative prompt(s) separated by commas.",
    )
    parser.add_argument(
        "--classes", "-c",
        default="wooden bridge,tree,water",
        help="Comma-separated class list for segmentation mode.",
    )
    parser.add_argument(
        "--url", "-u",
        default="http://localhost:8080",
        help="Base URL of the ReSiReg server.",
    )
    parser.add_argument(
        "--output", "-o",
        default="resireg_output.png",
        help="Path to write the output image PNG.",
    )
    parser.add_argument(
        "--model",
        default="SimonSchwaiger/resireg_mini",
        help="Model name to pass to the server.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the full JSON response to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    prompt_text = build_prompt(args)

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(args.image)}},
                ],
            }
        ],
    }

    url = f"{args.url.rstrip('/')}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"Sending request to {url} ...")
    print(f"Mode: {args.mode} | Prompt: {prompt_text}")

    try:
        with urllib.request.urlopen(req) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"Server returned HTTP {e.code}:", file=sys.stderr)
        print(e.read().decode("utf-8"), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    if args.show_json:
        print(json.dumps(response, indent=2))

    try:
        content = response["choices"][0]["message"]["content"]
        # The server may return the result as a JSON-encoded string.
        if isinstance(content, str):
            content = json.loads(content)
        result_b64 = content["result_image_base64"]
        # Strip data URI prefix if present
        if "," in result_b64:
            result_b64 = result_b64.split(",", 1)[1]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Unexpected response structure: {e}", file=sys.stderr)
        print(json.dumps(response, indent=2), file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.write_bytes(base64.b64decode(result_b64))
    print(f"Saved {content.get('mode', args.mode)} result to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
