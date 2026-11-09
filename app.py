"""Idol-Imagine-Fast: Text-to-3D & Image-to-3D (Gaussian splats) with Z-Image-Turbo and BRIA-RMBG-2.0.
Model code lives in model.py.
"""

import os
import random
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import spaces  # Hugging Face ZeroGPU; imported before the model so its CUDA placement is deferred
import torch
from diffusers import DiffusionPipeline
from gradio_client import Client, handle_file
from PIL import Image

import model  # noqa: E402  (after spaces)

# --- Environment Setup ---
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn_2"

# The splat viewer opens every object at orbit radius 5; this factor zooms it in on load (5.5/10 zoom instead of 3/10).
VIEWER_ZOOM = 5.5 / 3


# ==== PROJECT ==================================================================
NAME = "Idol-Imagine-Fast"
TAGLINE = (
    "<strong>One image or text prompt in, a 3D object out.</strong> Idol-Imagine-Fast turns a single picture or prompt "
    "into a 3D Gaussian splat. Generate from text, pick an example below, or <em>upload your own.</em>"
)
LINKS = {
    "app-code": "https://github.com/PRITHIVSAKTHIUR/idol-imagine-fast",
    "idol": "https://huggingface.co/speridlabs/idol",
}
# ===============================================================================

ROOT = Path(__file__).parent

DEFAULT_GAUSSIANS = (
    "262,144"
    if "262,144" in model.GAUSSIANS
    else next(
        (label for label, count in model.GAUSSIANS.items() if count == 262144),
        next(iter(model.GAUSSIANS)),
    )
)


# ==== MODELS (Z-Image-Turbo & BRIA-RMBG-2.0) ===================================
print("Initializing models...")
print("Loading Z-Image-Turbo...")
try:
    z_pipe = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_pipe.to(device)
    print("Z-Image-Turbo loaded.")
except Exception as e:
    print(f"Failed to load Z-Image-Turbo: {e}")
    z_pipe = None

rmbg_client = Client("briaai/BRIA-RMBG-2.0")


def remove_background(input: Image.Image) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        input = input.convert("RGB")
        input.save(f.name)
        output = rmbg_client.predict(handle_file(f.name), api_name="/image")[0][0]
        output = Image.open(output).convert("RGBA")
        return output


def preprocess_image(input: Image.Image) -> Image.Image:
    """Preprocess the input image: remove bg, crop, resize."""
    if input is None:
        return None

    has_alpha = False
    if input.mode == "RGBA":
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize(
            (int(input.width * scale), int(input.height * scale)),
            Image.Resampling.LANCZOS,
        )
    if has_alpha:
        output = input
    else:
        output = remove_background(input)

    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if bbox.size == 0:
        return output
    bbox = (
        np.min(bbox[:, 1]),
        np.min(bbox[:, 0]),
        np.max(bbox[:, 1]),
        np.max(bbox[:, 0]),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output


@spaces.GPU(size="xlarge")
def generate_txt2img(prompt: str, progress=gr.Progress(track_tqdm=True)):
    """Generate Image using Z-Image-Turbo."""
    if z_pipe is None:
        raise gr.Error("Z-Image-Turbo model failed to load.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(42)

    progress(0.1, desc="Generating Text-to-Image...")
    try:
        result = z_pipe(
            prompt=prompt,
            negative_prompt=None,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        )
        return result.images[0]
    except Exception as e:
        raise gr.Error(f"Z-Image Generation failed: {str(e)}")


# ==== IDOL 3D INFERENCE ========================================================
@spaces.GPU(duration=120, size="xlarge")
def infer(image, mode, steps, cfg_scale, rescale_t, seed, randomize, gaussians):
    if image is None:
        raise gr.Error("Upload an image, generate from text, or pick an example first.")
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    ply, prepared = model.generate(
        image,
        model.MODES[mode],
        int(steps),
        cfg_scale,
        rescale_t,
        int(seed),
        model.GAUSSIANS[gaussians],
    )
    return ply, prepared, seed


def meta(gaussians, steps):
    return f'<div class="sl-meta"><span>{gaussians} Gaussians · {int(steps)} steps</span></div>'


def run(image, mode, steps, cfg_scale, rescale_t, seed, randomize, gaussians):
    ply, _, seed = infer(image, mode, steps, cfg_scale, rescale_t, seed, randomize, gaussians)
    return ply, seed, meta(gaussians, steps)


# ==== STYLING & VIEWER JS ======================================================
CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 0 32px !important; }
footer { display: none !important; }
.sl-head { padding: 24px 0 18px; border-bottom: 2px solid rgba(128,128,128,0.2); margin-bottom: 12px; }
.sl-head h1 { margin: 0; font-size: clamp(28px, 3.2vw, 44px); font-weight: 700; line-height: 1.1; letter-spacing: -0.02em; }
.sl-row { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px 40px; }
.sl-head p { margin: 12px 0 0; max-width: 68ch; font-size: 17px; line-height: 1.5; }
.sl-head p em { color: var(--color-accent); font-style: normal; font-weight: 600; }
.sl-links { display: flex; gap: 10px; }
.sl-links a {
  display: inline-flex; align-items: center; height: 38px; padding: 0 18px;
  border-radius: 6px; border: 2px solid var(--color-accent); background: transparent;
  color: inherit !important; font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
  text-decoration: none; transition: background .15s, color .15s;
}
.sl-links a:hover { background: var(--color-accent); color: #000000 !important; }
.sl-meta {
  display: flex; gap: 16px; white-space: nowrap; padding-top: 6px;
  font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 13px;
}
.sl-section {
  border-top: 2px solid rgba(128,128,128,0.2); padding-top: 24px !important; margin-top: 24px;
  font-family: 'IBM Plex Mono', ui-monospace, monospace; text-transform: uppercase;
  letter-spacing: .1em; font-size: 14px; font-weight: 700;
}
#generate { height: 52px; font-size: 17px; font-weight: 600; }
#result { background: #0c0c0c; }
#result img { object-fit: contain; }
@media (max-width: 760px) {
  .sl-head { padding-top: 20px; }
  #sl-meta-spacer { display: none !important; }
}
"""

# Ensures file requests carry session cookies and zooms the 3D splat viewer on load.
VIEWER_JS = """() => {
  const VIEWER_ZOOM = VIEWER_ZOOM_VALUE;
  const originalFetch = window.fetch.bind(window);
  window.fetch = (resource, options) => {
    const url = typeof resource === 'string' ? resource : (resource && resource.url) || '';
    if (url.includes('/gradio_api/file=')) options = Object.assign({}, options, { credentials: 'same-origin' });
    const canvas = document.querySelector('#result canvas');
    if (canvas && url.split('?')[0].endsWith('.ply')) {
      const startRadius = 5, targetRadius = startRadius / VIEWER_ZOOM;
      const wheelScale = 0.025 * (0.1 + 0.9 * (startRadius - 0.1) / 29.9);
      canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: (targetRadius - startRadius) / wheelScale, cancelable: true }));
    }
    return originalFetch(resource, options);
  };
}""".replace("VIEWER_ZOOM_VALUE", f"{VIEWER_ZOOM:.6f}")

PHONE_HEAD = """<style>@media (max-width: 760px) {
  .gradio-container { padding: 0 16px !important; }
  main.fillable.app { padding-left: 0 !important; padding-right: 0 !important; }
}</style>"""

LINK_HTML = "".join(f'<a href="{u}" target="_blank">{t}</a>' for t, u in LINKS.items())
HEAD = f"""
<div class="sl-head">
  <div class="sl-row">
    <h1>{NAME}</h1>
    <div class="sl-links">{LINK_HTML}</div>
  </div>
  <p>{TAGLINE}</p>
</div>
"""

with gr.Blocks(title=NAME, head=PHONE_HEAD) as demo:
    gr.HTML(HEAD)
    with gr.Row(equal_height=True, elem_id="sl-main"):
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("Text-to-Image-3D"):
                    txt_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="eg. A Plane 3D",
                        lines=2,
                    )
                    btn_gen_img = gr.Button("1.Generate Image", variant="primary")
                with gr.Tab("Image-to-3D"):
                    gr.Markdown("Upload an image directly if you have one.")

            image = gr.Image(
                value=None,
                type="pil",
                image_mode="RGBA",
                label="Input image",
                sources=["upload", "clipboard"],
                height=360,
                elem_id="sl-input",
            )
            mode = gr.Radio(list(model.MODES), value=next(iter(model.MODES)), label="Input type")
            run_btn = gr.Button("2.Generate 3D", variant="primary", elem_id="generate")
            with gr.Accordion("Settings", open=False, elem_id="sl-settings"):
                gaussians = gr.Dropdown(
                    list(model.GAUSSIANS),
                    value=DEFAULT_GAUSSIANS,
                    label="Gaussians",
                )
                steps = gr.Slider(10, 100, value=50, step=1, label="Steps")
                cfg_scale = gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="Guidance scale")
                rescale_t = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Time shift")
                seed = gr.Number(value=42, precision=0, label="Seed")
                randomize = gr.Checkbox(value=False, label="Random seed")
        with gr.Column(scale=7):
            result = gr.Model3D(
                show_label=False,
                elem_id="result",
                height="min(640px, 92vw)",
                clear_color=(0.05, 0.05, 0.05, 1.0),
            )
    # below the viewer, outside the main row so the input column ends where the viewer ends
    with gr.Row():
        gr.Column(scale=5, min_width=0, elem_id="sl-meta-spacer")
        with gr.Column(scale=7):
            info = gr.HTML("")

    gr.HTML('<div class="sl-section">Examples — click to load an image or prompt</div>')

    gr.Examples(
        examples=[
            ["example-images/A (1).webp"],
            ["example-images/A (2).webp"],
            ["example-images/A (3).webp"],
            ["example-images/A (4).webp"],
            ["example-images/A (5).webp"],
            ["example-images/A (6).webp"],
            ["example-images/A (7).webp"],
            ["example-images/A (8).webp"],
            ["example-images/A (9).webp"],
            ["example-images/A (10).webp"],
            ["example-images/A (11).webp"],
            ["example-images/A (12).webp"],
            ["example-images/A (13).webp"],
            ["example-images/A (14).webp"],
            ["example-images/A (15).webp"],
            ["example-images/A (16).webp"],
            ["example-images/A (17).webp"],
            ["example-images/A (18).webp"],
            ["example-images/A (19).webp"],
            ["example-images/A (20).webp"],
            ["example-images/A (21).webp"],
            ["example-images/A (22).webp"],
            ["example-images/A (23).webp"],
            ["example-images/A (24).webp"],
            ["example-images/A (25).webp"],
            ["example-images/A (26).webp"],
            ["example-images/A (27).webp"],
            ["example-images/A (28).webp"],
            ["example-images/A (29).webp"],
            ["example-images/A (30).webp"],
            ["example-images/A (31).webp"],
            ["example-images/A (32).webp"],
            ["example-images/A (33).webp"],
            ["example-images/A (34).webp"],
            ["example-images/A (35).webp"],
            ["example-images/A (36).webp"],
            ["example-images/A (37).webp"],
            ["example-images/A (38).webp"],
            ["example-images/A (39).webp"],
            ["example-images/A (40).webp"],
            ["example-images/A (41).webp"],
            ["example-images/A (42).webp"],
            ["example-images/A (43).webp"],
            ["example-images/A (44).webp"],
            ["example-images/A (45).webp"],
            ["example-images/A (46).webp"],
            ["example-images/A (47).webp"],
            ["example-images/A (48).webp"],
            ["example-images/A (49).webp"],
            ["example-images/A (50).webp"],
            ["example-images/A (51).webp"],
            ["example-images/A (52).webp"],
            ["example-images/A (53).webp"],
            ["example-images/A (54).webp"],
            ["example-images/A (55).webp"],
            ["example-images/A (56).webp"],
            ["example-images/A (57).webp"],
            ["example-images/A (58).webp"],
            ["example-images/A (59).webp"],
            ["example-images/A (60).webp"],
            ["example-images/A (61).webp"],
            ["example-images/A (62).webp"],
            ["example-images/A (63).webp"],
            ["example-images/A (64).webp"],
            ["example-images/A (65).webp"],
            ["example-images/A (66).webp"],
            ["example-images/A (67).webp"],
            ["example-images/A (68).webp"],
            ["example-images/A (69).webp"],
            ["example-images/A (70).webp"],
            ["example-images/A (71).webp"],
        ],
        inputs=[image],
        label="Image Examples [image-to-3d]",
    )

    gr.Examples(
        examples=[
            ["A Cat 3D model"],
            ["A realistic Cat 3D model"],
            ["A cartoon Cat 3D model"],
            ["A low poly Cat 3D"],
            ["A cyberpunk Cat 3D"],
            ["A robotic Cat 3D"],
            ["A fluffy Cat 3D"],
            ["A fantasy Cat 3D creature"],
            ["A stylized Cat 3D"],
            ["A Cat 3D sculpture"],
            ["A Plane 3D model"],
            ["A commercial Plane 3D"],
            ["A fighter jet Plane 3D"],
            ["A low poly Plane 3D"],
            ["A vintage Plane 3D"],
            ["A futuristic Plane 3D"],
            ["A cargo Plane 3D"],
            ["A private jet Plane 3D"],
            ["A toy Plane 3D"],
            ["A realistic Plane 3D"],
            ["A Car 3D model"],
            ["A sports Car 3D"],
            ["A luxury Car 3D"],
            ["A low poly Car 3D"],
            ["A racing Car 3D"],
            ["A cyberpunk Car 3D"],
            ["A vintage Car 3D"],
            ["A futuristic Car 3D"],
            ["A SUV Car 3D"],
            ["A electric Car 3D"],
            ["A Shoe 3D model"],
            ["A sneaker Shoe 3D"],
            ["A running Shoe 3D"],
            ["A leather Shoe 3D"],
            ["A high heel Shoe 3D"],
            ["A boot Shoe 3D"],
            ["A low poly Shoe 3D"],
            ["A futuristic Shoe 3D"],
            ["A sports Shoe 3D"],
            ["A casual Shoe 3D"],
            ["A Chair 3D model"],
            ["A Table 3D model"],
            ["A Sofa 3D model"],
            ["A Lamp 3D model"],
            ["A Watch 3D model"],
            ["A Backpack 3D model"],
            ["A Drone 3D model"],
            ["A Robot 3D model"],
            ["A Smartphone 3D model"],
            ["A Headphones 3D model"],
            ["A House 3D model"],
            ["A Skyscraper 3D model"],
            ["A Bridge 3D model"],
            ["A Castle 3D model"],
            ["A Spaceship 3D model"],
            ["A Rocket 3D model"],
            ["A Satellite 3D model"],
            ["A Tank 3D model"],
            ["A Motorcycle 3D model"],
            ["A Bicycle 3D model"],
        ],
        inputs=[txt_prompt],
        label="3D Prompt Examples [text-to-3d]",
    )

    inputs = [image, mode, steps, cfg_scale, rescale_t, seed, randomize, gaussians]

    btn_gen_img.click(
        generate_txt2img,
        inputs=[txt_prompt],
        outputs=[image],
    ).then(
        preprocess_image,
        inputs=[image],
        outputs=[image],
    )

    image.upload(
        preprocess_image,
        inputs=[image],
        outputs=[image],
    )

    run_btn.click(run, inputs, [result, seed, info], show_progress_on=result)

if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Citrus(),
        css=CSS,
        js=VIEWER_JS,
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
    )