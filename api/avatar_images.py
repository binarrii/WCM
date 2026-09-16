"""Bound and normalize uploaded profile images before storing them."""

import io

from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_AVATAR_BYTES = 5 * 1024 * 1024
MAX_AVATAR_PIXELS = 25_000_000
AVATAR_SIZE = 256


def normalize_avatar(content: bytes) -> bytes:
    if len(content) > MAX_AVATAR_BYTES:
        raise HTTPException(413, "头像图片不能超过 5 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(422, "请选择 JPG、PNG 或 WebP 图片")
            if image.width * image.height > MAX_AVATAR_PIXELS:
                raise HTTPException(413, "头像图片像素过大，请先缩小到 2500 万像素以内")
            if getattr(image, "n_frames", 1) != 1:
                raise HTTPException(422, "请选择静态图片作为头像")
            image.load()
            square = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGBA"),
                (AVATAR_SIZE, AVATAR_SIZE),
                method=Image.Resampling.LANCZOS,
            )
            # A fresh canvas drops EXIF/location metadata and flattens transparency.
            canvas = Image.new("RGB", square.size, "#f6f7fb")
            canvas.paste(square, mask=square.getchannel("A"))
            output = io.BytesIO()
            canvas.save(output, format="JPEG", quality=85, optimize=True)
            normalized = output.getvalue()
            if len(normalized) > 65535:
                raise HTTPException(422, "图片内容过于复杂，请换一张图片")
            return normalized
    except Image.DecompressionBombError:
        raise HTTPException(413, "头像图片像素过大，请先缩小图片")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(422, "图片无法读取，请选择有效的 JPG、PNG 或 WebP 图片")
