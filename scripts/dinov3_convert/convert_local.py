"""Convert the LOCAL DINOv3 ViT-B/16 .pth into HuggingFace format.

Reuses the official transformers conversion logic (convert.py in this folder),
but loads weights from the already-downloaded local .pth instead of the gated
Hub repo. Keeps the official reference-output validation so we know the
conversion is bit-for-bit correct before trusting any LAM eval.
"""

import argparse
import os
import sys

import torch

# Reuse the official mapping/config/validation helpers verbatim.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import (  # noqa: E402
    convert_old_keys_to_new_keys,
    get_dinov3_config,
    get_image_processor,
    get_transform,
    prepare_img,
    split_qkv,
)

from transformers import DINOv3ViTModel  # noqa: E402


# Reference cls/patch outputs for the canonical COCO image (from the official script).
EXPECTED = {
    "vitb16_lvd1689m_cls": [1.034643, -0.180609, -0.341018, -0.066376, -0.011383],
    "vitb16_lvd1689m_patch": [-0.082523, -0.456272, -0.728029, -0.430680, -0.152880],
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Local original DINOv3 .pth file")
    ap.add_argument("--save-dir", required=True, help="Output dir for HF files")
    ap.add_argument("--model-name", default="vitb16_lvd1689m")
    ap.add_argument("--resize-size", type=int, default=224,
                    help="Validation resize (must be 224 to match reference outputs)")
    args = ap.parse_args()

    model_name = args.model_name
    config = get_dinov3_config(model_name)
    model = DINOv3ViTModel(config).eval()

    # --- ONLY change vs the official script: load from local .pth ---
    original_state_dict = torch.load(args.pth, map_location="cpu")
    if isinstance(original_state_dict, dict) and "state_dict" in original_state_dict:
        original_state_dict = original_state_dict["state_dict"]

    original_state_dict = split_qkv(original_state_dict)
    original_keys = list(original_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(original_keys)

    converted_state_dict = {}
    for key in original_keys:
        new_key = new_keys[key]
        weight_tensor = original_state_dict[key]
        if "bias_mask" in key or "attn.k_proj.bias" in key or "local_cls_norm" in key:
            continue
        if key.startswith("projectors."):
            continue
        if "embeddings.mask_token" in new_key:
            weight_tensor = weight_tensor.unsqueeze(1)
        if "inv_freq" in new_key:
            continue
        # NOTE: transformers 5.2.0 DINOv3ViTModel uses a flat layout (no `model.`
        # prefix), unlike the main-branch conversion script this is adapted from.
        converted_state_dict[new_key] = weight_tensor

    model.load_state_dict(converted_state_dict, strict=True)
    model = model.eval()
    print("[ok] state_dict loaded strict=True")

    # --- Reference-output validation on the canonical COCO image ---
    transform = get_transform(args.resize_size)
    image_processor = get_image_processor(args.resize_size)
    image = prepare_img()

    original_pixel_values = transform(image).unsqueeze(0)
    inputs = image_processor(image, return_tensors="pt")
    torch.testing.assert_close(original_pixel_values, inputs["pixel_values"], atol=1e-6, rtol=1e-6)
    print("[ok] preprocessing matches torchvision transform")

    out = model(**inputs)
    cls = out.pooler_output[0, :5].tolist()
    patch = out.last_hidden_state[:, config.num_register_tokens + 1:][0, 0, :5].tolist()
    print("cls  actual  :", [round(x, 6) for x in cls])
    print("cls  expected:", EXPECTED[f"{model_name}_cls"])
    torch.testing.assert_close(torch.tensor(cls), torch.tensor(EXPECTED[f"{model_name}_cls"]),
                               atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(torch.tensor(patch), torch.tensor(EXPECTED[f"{model_name}_patch"]),
                               atol=1e-3, rtol=1e-3)
    print("[ok] forward pass matches reference outputs (atol=1e-3)")

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    image_processor.save_pretrained(args.save_dir)
    print(f"[done] HF model saved to {args.save_dir}")


if __name__ == "__main__":
    main()
