"""
Hypertile module for splitting attention layers in SD-1.5 U-Net and SD-1.5 VAE
Warn : The patch works well only if the input image has a width and height that are multiples of 128
Author : @tfernd Github : https://github.com/tfernd/HyperTile
"""

from __future__ import annotations
from typing import Callable
from typing_extensions import Literal

import logging
from functools import wraps
from contextlib import contextmanager

import math
import torch.nn as nn
import random

from einops import rearrange

# TODO add SD-XL layers
DEPTH_LAYERS = {
    0: [
        # SD 1.5 U-Net (diffusers)
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "input_blocks.1.1.transformer_blocks.0.attn1",
        "input_blocks.2.1.transformer_blocks.0.attn1",
        "output_blocks.9.1.transformer_blocks.0.attn1",
        "output_blocks.10.1.transformer_blocks.0.attn1",
        "output_blocks.11.1.transformer_blocks.0.attn1",
        # SD 1.5 VAE
        "decoder.mid_block.attentions.0",
    ],
    1: [
        # SD 1.5 U-Net (diffusers)
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "input_blocks.4.1.transformer_blocks.0.attn1",
        "input_blocks.5.1.transformer_blocks.0.attn1",
        "output_blocks.6.1.transformer_blocks.0.attn1",
        "output_blocks.7.1.transformer_blocks.0.attn1",
        "output_blocks.8.1.transformer_blocks.0.attn1",
    ],
    2: [
        # SD 1.5 U-Net (diffusers)
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "input_blocks.7.1.transformer_blocks.0.attn1",
        "input_blocks.8.1.transformer_blocks.0.attn1",
        "output_blocks.3.1.transformer_blocks.0.attn1",
        "output_blocks.4.1.transformer_blocks.0.attn1",
        "output_blocks.5.1.transformer_blocks.0.attn1",
    ],
    3: [
        # SD 1.5 U-Net (diffusers)
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "middle_block.1.transformer_blocks.0.attn1",
    ],
}
RNG_INSTANCE = random.Random()

def random_divisor(value: int, min_value: int, /, max_options: int = 1) -> int:
    """
    Returns a random divisor of value that
        x * min_value <= value
    if max_options is 1, the behavior is deterministic
    """
    min_value = min(min_value, value)

    # All big divisors of value (inclusive)
    divisors = [i for i in range(min_value, value + 1) if value % i == 0] # divisors in small -> big order

    ns = [value // i for i in divisors[:max_options]]  # has at least 1 element # big -> small order

    idx = RNG_INSTANCE.randint(0, len(ns) - 1)

    return ns[idx]

def set_hypertile_seed(seed: int) -> None:
    RNG_INSTANCE.seed(seed)
    
def largest_tile_size_available(width:int, height:int) -> int:
    """
    Calculates the largest tile size available for a given width and height
    Tile size is always a power of 2
    """
    gcd = math.gcd(width, height)
    largest_tile_size_available = 1
    while gcd % (largest_tile_size_available * 2) == 0:
        largest_tile_size_available *= 2
    return largest_tile_size_available

@contextmanager
def split_attention(
    layer: nn.Module,
    /,
    aspect_ratio: float,  # width/height
    tile_size: int = 256,  # 128 for VAE
    swap_size: int = 2,  # 1 for VAE
    *,
    disable: bool = False,
    max_depth: Literal[0, 1, 2, 3] = 0,  # ! Try 0 or 1
    scale_depth: bool = False,  # scale the tile-size depending on the depth
):
    # Hijacks AttnBlock from ldm and Attention from diffusers

    if disable:
        logging.info(f"Attention for {layer.__class__.__qualname__} not splitted")
        yield
        return

    latent_tile_size = max(32, tile_size) // 8

    def self_attn_forward(forward: Callable, depth: int, layer_name: str, module: nn.Module) -> Callable:
        @wraps(forward)
        def wrapper(*args, **kwargs):
            x = args[0]

            # VAE
            if x.ndim == 4:
                b, c, h, w = x.shape

                nh = random_divisor(h, latent_tile_size, swap_size)
                nw = random_divisor(w, latent_tile_size, swap_size)

                if nh * nw > 1:
                    x = rearrange(x, "b c (nh h) (nw w) -> (b nh nw) c h w", nh=nh, nw=nw) # split into nh * nw tiles

                out = forward(x, *args[1:], **kwargs)

                if nh * nw > 1:
                    out = rearrange(out, "(b nh nw) c h w -> b c (nh h) (nw w)", nh=nh, nw=nw)

            # U-Net
            else:
                hw = x.size(1)
                h, w = round(math.sqrt(hw * aspect_ratio)), round(math.sqrt(hw / aspect_ratio))
                # h*w should be equal to hw
                assert h * w == hw, f"Invalid aspect ratio {aspect_ratio} for input of shape {x.shape}"

                factor = 2**depth if scale_depth else 1
                nh = random_divisor(h, latent_tile_size * factor, swap_size)
                nw = random_divisor(w, latent_tile_size * factor, swap_size)

                module._split_sizes_hypertile.append((nh, nw))  # type: ignore

                if nh * nw > 1:
                    x = rearrange(x, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)

                out = forward(x, *args[1:], **kwargs)

                if nh * nw > 1:
                    out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                    out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)

            return out

        return wrapper

    # Handle hijacking the forward method and recovering afterwards
    try:
        for depth in range(max_depth + 1):
            for layer_name, module in layer.named_modules():
                if any(layer_name.endswith(try_name) for try_name in DEPTH_LAYERS[depth]):
                    # print input shape for debugging
                    logging.info(f"HyperTile hijacking attention layer at depth {depth}: {layer_name}")

                    # save original forward for recovery later
                    setattr(module, "_original_forward_hypertile", module.forward)
                    setattr(module, "forward", self_attn_forward(module.forward, depth, layer_name, module))

                    setattr(module, "_split_sizes_hypertile", [])
        yield
    finally:
        for layer_name, module in layer.named_modules():
            # remove hijack
            if hasattr(module, "_original_forward_hypertile"):
                if module._split_sizes_hypertile:
                    logging.debug(f"layer {layer_name} splitted with ({module._split_sizes_hypertile})")

                setattr(module, "forward", module._original_forward_hypertile)
                del module._original_forward_hypertile
                del module._split_sizes_hypertile