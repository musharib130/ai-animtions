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
    "photo, photorealistic"
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


def generate_character(request: CharacterRequest, pipe=None) -> list[Path]:
    import torch

    if pipe is None:
        pipe = load_pipeline()

    full_prompt = build_prompt(request.prompt, request.style)
    base_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
    slug = re.sub(r"[^a-z0-9]+", "-", request.prompt.lower()).strip("-")[:40] or "character"

    request.out_dir.mkdir(parents=True, exist_ok=True)
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

        path = request.out_dir / f"{slug}_{request.style}_{seed}.png"
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
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        saved.append(path)
        print(f"Saved {path}")

    return saved
