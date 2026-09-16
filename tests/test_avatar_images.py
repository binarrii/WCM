import io
import struct
import zlib

import pytest
from fastapi import HTTPException
from PIL import Image

from api.avatar_images import MAX_AVATAR_BYTES, normalize_avatar


def image_bytes(format="PNG", size=(320, 180), color="red", **options):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format=format, **options)
    return stream.getvalue()


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_avatar_normalizes_supported_images_to_small_square(format):
    result = normalize_avatar(image_bytes(format))
    with Image.open(io.BytesIO(result)) as image:
        assert image.format == "JPEG"
        assert image.size == (256, 256)
        assert image.mode == "RGB"
    assert len(result) < 65536


def test_avatar_removes_exif_and_transparency():
    exif = Image.Exif()
    exif[270] = "private camera metadata"
    exif[274] = 6
    with Image.open(io.BytesIO(normalize_avatar(image_bytes("JPEG", exif=exif)))) as result:
        assert not result.getexif()
    stream = io.BytesIO()
    Image.new("RGBA", (40, 70), (255, 0, 0, 0)).save(stream, format="PNG")
    with Image.open(io.BytesIO(normalize_avatar(stream.getvalue()))) as result:
        assert min(result.getpixel((128, 128))) > 240


@pytest.mark.parametrize(
    "content",
    [b"", b"<svg xmlns='http://www.w3.org/2000/svg'/>", b"not an image", image_bytes("GIF")],
)
def test_avatar_rejects_invalid_and_unsupported_images(content):
    with pytest.raises(HTTPException) as error:
        normalize_avatar(content)
    assert error.value.status_code == 422


def test_avatar_rejects_oversized_files_and_pixel_dimensions():
    with pytest.raises(HTTPException) as error:
        normalize_avatar(b"x" * (MAX_AVATAR_BYTES + 1))
    assert error.value.status_code == 413
    content = image_bytes()
    header = struct.pack(">II", 6000, 6000) + content[24:29]
    large_header = (
        content[:16] + header + struct.pack(">I", zlib.crc32(b"IHDR" + header)) + content[33:]
    )
    with pytest.raises(HTTPException) as error:
        normalize_avatar(large_header)
    assert error.value.status_code == 413


def test_avatar_rejects_animated_images():
    stream = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(
        stream,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (20, 20), "blue")],
        duration=100,
    )
    with pytest.raises(HTTPException) as error:
        normalize_avatar(stream.getvalue())
    assert error.value.status_code == 422
