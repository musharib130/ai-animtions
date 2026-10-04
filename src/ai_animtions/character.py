"""Generate animation-ready character images from a text prompt using SDXL 1.0."""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

STYLES = {
    "cartoon": "2D cartoon character design, clean bold outlines, flat cel shading, vibrant colors",
    "3d": "stylized 3D animated film character, soft studio lighting, smooth shading, appealing proportions",
    "anime": "anime character design, clean lineart, cel shading, vibrant colors",
}

# Framing that keeps characters usable for animation: one figure, full body,
# neutral pose, plain background that is easy to cut out.
TEMPLATE = (
    "{style}, full body character design of {prompt}, standing in a neutral pose, "
    "front view, centered, single character, plain white background, highly detailed"
)

NEGATIVE_PROMPT = (
    "blurry, lowres, cropped, out of frame, cut off feet, extra limbs, extra fingers, "
    "deformed hands, multiple characters, text, watermark, signature, busy background, "
    "photo, photorealistic, gun, weapon, pistol, holding object, cigarette, smoking, jetpack"
)


@dataclass
class CharacterRequest:
    prompt: str
    style: str = "cartoon"
    seed: int | None = None
    count: int = 1
    steps: int = 30
    guidance: float = 7.0
    width: int = 832
    height: int = 1216
    out_dir: Path = Path("outputs/characters")
    remove_bg: bool = True


# Background removal models (rembg, runs on CPU). isnet-anime is trained on
# drawn characters; 3D renders need the general-purpose model.
BG_MODELS = {
    "cartoon": "isnet-anime",
    "anime": "isnet-anime",
    "3d": "isnet-general-use",
}


# Each output folder groups files by type; a file keeps the same name in each.
WITH_BG_DIR, NO_BG_DIR, JSON_DIR = "with_bg", "no_bg", "json"


def output_paths(out_dir: Path, name: str) -> tuple[Path, Path, Path]:
    """Return (with_bg .png, no_bg .png, .json) paths for name, creating the folders."""
    paths = (
        out_dir / WITH_BG_DIR / f"{name}.png",
        out_dir / NO_BG_DIR / f"{name}.png",
        out_dir / JSON_DIR / f"{name}.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def build_prompt(prompt: str, style: str) -> str:
    return TEMPLATE.format(style=STYLES[style], prompt=prompt)


def load_pipeline():
    print("Loading libraries (first run can take a minute)...", flush=True)
    import torch
    from diffusers import StableDiffusionXLPipeline

    print(f"Loading {MODEL_ID} (downloads ~7 GB on first run)...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    # Moves model parts between GPU and RAM as needed so SDXL fits in 8 GB VRAM.
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def remove_background(image_path: Path, out_path: Path, style: str = "cartoon", session=None) -> Path:
    """Save a transparent-background copy of image_path to out_path."""
    from PIL import Image
    from rembg import new_session, remove

    if session is None:
        session = new_session(BG_MODELS[style])

    cutout = remove(Image.open(image_path), session=session, post_process_mask=True)
    cutout.save(out_path)
    return out_path


def generate_character(request: CharacterRequest, pipe=None) -> list[Path]:
    import torch

    if pipe is None:
        pipe = load_pipeline()

    full_prompt = build_prompt(request.prompt, request.style)
    base_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
    slug = re.sub(r"[^a-z0-9]+", "-", request.prompt.lower()).strip("-")[:40] or "character"

    bg_session = None
    if request.remove_bg:
        from rembg import new_session

        print(f"Loading background removal model {BG_MODELS[request.style]}...", flush=True)
        bg_session = new_session(BG_MODELS[request.style])

    saved = []
    for i in range(request.count):
        seed = base_seed + i
        image = pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance,
            width=request.width,
            height=request.height,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        path, nobg_path, json_path = output_paths(request.out_dir, f"{slug}_{request.style}_{seed}")
        image.save(path)

        # Sidecar with everything needed to regenerate the same character later.
        meta = asdict(request) | {
            "seed": seed,
            "full_prompt": full_prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "model": MODEL_ID,
            "out_dir": str(request.out_dir),
        }
        meta.pop("count")
        json_path.write_text(json.dumps(meta, indent=2))
        saved.append(path)
        print(f"Saved {path}")

        if bg_session is not None:
            remove_background(path, nobg_path, session=bg_session)
            saved.append(nobg_path)
            print(f"Saved {nobg_path}")

    return saved
