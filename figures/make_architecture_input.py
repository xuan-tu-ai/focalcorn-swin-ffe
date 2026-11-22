#!/usr/bin/env python3
"""Crop the input thumbnail shown in F2 (architecture) from one FFE image.

Run: python figures/make_architecture_input.py    (from the repository root)
Takes the square centre crop of one Highly Fresh image, at native resolution, and
writes F2_input.jpg next to this script; F2_architecture.tex then includes it.
The image choice is illustrative and has no bearing on any result.
"""
from pathlib import Path

from PIL import Image

SRC = Path("data/Rastrelliger Faughni - Highly Fresh/IMG_20191014_065738.jpg")
OUT = Path(__file__).with_name("F2_input.jpg")

im = Image.open(SRC).convert("RGB")
w, h = im.size
s = min(w, h)
left, top = (w - s) // 2, (h - s) // 2
im.crop((left, top, left + s, top + s)).save(OUT, quality=95)
print(f"{SRC} ({w}x{h}) -> {OUT} ({s}x{s})")
