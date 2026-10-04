import os
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

src = Path("/home/akif/Gen AI/Assignment_1/Generative_AI/data/images")
dst = Path("/home/akif/Gen AI/Assignment_1/Generative_AI/data/images_128")
dst.mkdir(parents=True, exist_ok=True)

def resize_one(p):
    with Image.open(p) as im:
        im.convert("RGB").resize((128,128), Image.Resampling.BICUBIC).save(dst / p.name, quality=95)

with ThreadPoolExecutor(max_workers=8) as ex:
    ex.map(resize_one, src.glob("*.jpg"))