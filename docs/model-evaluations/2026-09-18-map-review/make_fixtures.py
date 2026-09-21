import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/tmp/wcm-map-eval")
font_path = "/System/Library/Fonts/STHeiti Light.ttc"
font = ImageFont.truetype(font_path, 22)
small = ImageFont.truetype(font_path, 17)
title = ImageFont.truetype(font_path, 30)
polys = json.loads((ROOT / "china.geojson").read_text())["features"][0]["geometry"]["coordinates"]


def canvas(bounds, heading):
    image = Image.new("RGB", (1080, 800), "#e6f3fa")
    d = ImageDraw.Draw(image)
    west, b, r, t = bounds

    def xy(p):
        return (40 + (p[0] - west) / (r - west) * 1000, 100 + (t - p[1]) / (t - b) * 630)

    for lon in range(math.ceil(west / 5) * 5, math.floor(r) + 1, 5):
        x, _ = xy((lon, b))
        d.line((x, 100, x, 730), fill="#c3d5df")
        d.text((x, 745), str(lon) + "°E", font=small, fill="#647781")
    for lat in range(math.ceil(b / 5) * 5, math.floor(t) + 1, 5):
        _, y = xy((west, lat))
        d.line((40, y, 1040, y), fill="#c3d5df")
    d.rectangle((0, 0, 1080, 82), fill="white")
    d.text((34, 23), heading, font=title, fill="#172737")
    return image, d, xy


def taiwan(ring):
    return all(119.3 < p[0] < 122.5 and 21.5 < p[1] < 25.8 for p in ring)


def china(name, mode):
    im, d, xy = canvas((72, 16, 137, 56), "中国全图")
    for poly in polys:
        ring = poly[0]
        if mode == "missing" and taiwan(ring):
            continue
        color = "#70a7e2" if mode == "foreign" and taiwan(ring) else "#eed096"
        d.polygon([xy(p) for p in ring], fill=color, outline="#716352", width=2)
    d.text(xy((98, 36)), "中国", font=title, fill="#272b31")
    d.rectangle((44, 675, 230, 725), fill="white", outline="#999")
    d.rectangle((55, 690, 78, 711), fill="#eed096")
    d.text((89, 687), "国别：中国", font=font, fill="#222")
    if mode == "foreign":
        d.rectangle((770, 640, 1035, 725), fill="white", outline="#999")
        d.rectangle((780, 691, 803, 713), fill="#70a7e2")
        d.text((814, 687), "国别：日本", font=font, fill="#222")
        d.text(xy((120, 26.5)), "台湾", font=font, fill="#222")
    im.save(ROOT / name)
    return im


full = china("china-complete.png", "complete")
china("china-without-taiwan.png", "missing")
china("china-taiwan-foreign.png", "foreign")
full.crop((70, 180, 530, 450)).resize((1080, 635)).save(ROOT / "china-crop.png")
for correct in (False, True):
    im, d, xy = canvas((121, 40, 142, 56), "中俄边境地区图")
    for poly in polys:
        d.polygon([xy(p) for p in poly[0]], fill="#eed096", outline="#716352", width=2)
    d.rectangle((0, 0, 1080, 82), fill="white")
    d.text((34, 23), "中俄边境地区图", font=title, fill="#172737")
    d.text(xy((124, 45)), "中国", font=title, fill="#334")
    d.text(xy((137, 52)), "俄罗斯", font=title, fill="#334")
    for point, current, old in [
        ((131.89, 43.12), "符拉迪沃斯托克", "海参崴"),
        ((127.54, 50.29), "布拉戈维申斯克", "海兰泡"),
    ]:
        x, y = xy(point)
        label = current + ("（" + old + "）" if correct else "")
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#344d78")
        box = d.textbbox((x + 12, y - 12), label, font=font)
        d.rectangle((box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3), fill="white")
        d.text((x + 12, y - 12), label, font=font, fill="#202a37")
    im.save(ROOT / ("russian-names-correct.png" if correct else "russian-names-missing.png"))
im, d, xy = canvas((-6, 41, 10, 52), "法国地图")
d.polygon(
    [
        xy(p)
        for p in [
            (-4.8, 48.4),
            (-1.9, 48.6),
            (-1.5, 49.7),
            (1.7, 50.8),
            (4, 50),
            (7.5, 48.8),
            (7.7, 43.8),
            (5.8, 43),
            (3, 42.4),
            (-1.8, 43.3),
            (-1.2, 46.3),
            (-4.8, 48.4),
        ]
    ],
    fill="#d9e4c6",
    outline="#53644c",
    width=3,
)
for point, label in [((2.35, 48.86), "巴黎"), ((4.84, 45.76), "里昂"), ((5.37, 43.30), "马赛")]:
    x, y = xy(point)
    d.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#253d31")
    d.text((x + 8, y - 12), label, font=font, fill="#253d31")
im.save(ROOT / "foreign-france.png")
im = Image.new("RGB", (1080, 800), "white")
d = ImageDraw.Draw(im)
d.text((70, 120), "地名词语练习", font=title, fill="#333")
d.text((70, 220), "符拉迪沃斯托克\n\n布拉戈维申斯克", font=font, fill="#333")
im.save(ROOT / "text-no-map.png")
cases = [
    {"image": name, "expected": expect}
    for name, expect in [
        ("china-complete.png", {}),
        ("china-without-taiwan.png", {"china_map_missing_region": 1}),
        ("china-taiwan-foreign.png", {"china_map_foreign_region": 1}),
        ("russian-names-missing.png", {"map_missing_chinese_name": 2}),
        ("russian-names-correct.png", {}),
        ("china-crop.png", {}),
        ("foreign-france.png", {}),
        ("text-no-map.png", {}),
    ]
]
(ROOT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
print("Created", len(cases), "map fixtures")
