import argparse
from pathlib import Path

from ai_animtions.character import STYLES, CharacterRequest, generate_character


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-animtions",
        description="Create an animation-ready character from a text prompt using SDXL 1.0.",
    )
    parser.add_argument("prompt", help='Character description, e.g. "a friendly pilot in a leather jacket"')
    parser.add_argument("--style", choices=STYLES, default="cartoon", help="Art style (default: cartoon)")
    parser.add_argument("--seed", type=int, help="Seed for reproducible results (default: random)")
    parser.add_argument("--count", type=int, default=1, help="Number of variations to generate (default: 1)")
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps (default: 30)")
    parser.add_argument("--guidance", type=float, default=7.0, help="Prompt adherence (default: 7.0)")
    parser.add_argument("--width", type=int, default=832, help="Image width (default: 832)")
    parser.add_argument("--height", type=int, default=1216, help="Image height (default: 1216)")
    parser.add_argument("--out", type=Path, default=Path("outputs/characters"), help="Output folder")
    parser.add_argument("--keep-bg", action="store_true", help="Skip background removal")
    args = parser.parse_args()

    generate_character(
        CharacterRequest(
            prompt=args.prompt,
            style=args.style,
            seed=args.seed,
            count=args.count,
            steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            out_dir=args.out,
            remove_bg=not args.keep_bg,
        )
    )
