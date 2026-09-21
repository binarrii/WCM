import asyncio
import base64
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path

from PIL import Image

from api import flags, parameter_store
from wcm_facerec.config import settings


async def main():
    folder = Path("/eval")
    await parameter_store.refresh()
    times = [3, 13, 143, 325, 535, 754, 1003, 1192.2, 1322, 1436]
    cases = []
    for second in times:
        name = f"travel-{second:g}.png"
        path = folder / name
        if not path.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(second),
                    "-i",
                    "http://10.252.25.251:18080/videos/20160202.mp4",
                    "-frames:v",
                    "1",
                    "-y",
                    str(path),
                ],
                check=True,
                timeout=90,
            )
        cases.append({"image": name, "seconds": second, "expected_ntd": 0})
    cases.append({"image": "ntd-positive.png", "expected_ntd": 1})
    report = {
        "model": settings.flags_model,
        "prompt": flags.build_prompt(),
        "source_task": "4da36fdd-a10f-4924-87d3-9a5e8ef33ddd",
        "runtime_sha256": {
            p: hashlib.sha256(Path("/app", p).read_bytes()).hexdigest()
            for p in ("api/flags.py", "src/wcm_facerec/object_detection_policy.py")
        },
        "cases": [],
    }
    for case in cases:
        image = Image.open(folder / case["image"]).convert("RGB")
        image.thumbnail((1080, 1080))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=settings.jpeg_quality)
        start = time.monotonic()
        result = dict(case, size=list(image.size))
        try:
            detections = await flags.detect(base64.b64encode(buf.getvalue()).decode())
            ntd = [d for d in detections if d.get("organization") == "新唐人"]
            result.update(
                detections=detections, actual_ntd=len(ntd), passed=len(ntd) == case["expected_ntd"]
            )
        except Exception as exc:
            result.update(error=type(exc).__name__ + ": " + str(exc), passed=False)
        result["elapsed_seconds"] = round(time.monotonic() - start, 2)
        report["cases"].append(result)
        (folder / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            json.dumps({k: v for k, v in result.items() if k != "detections"}, ensure_ascii=False),
            flush=True,
        )
    print("PASSED", sum(c["passed"] for c in report["cases"]), len(cases), flush=True)


asyncio.run(main())
