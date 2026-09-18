"""Exercise the production object detector on explicit local fixtures, without creating tasks.

The fixture folder contains cases.json: image filenames and expected target counts.
Only pixels and the configured production prompt are sent to the model, never labels.
Use --refresh-parameters inside an API container to read existing gateway credentials.
"""

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

from PIL import Image

from api import flags, parameter_store
from wcm_facerec.config import settings


async def evaluate(folder, refresh_parameters):
    if refresh_parameters:
        await parameter_store.refresh()
    cases = json.loads((folder / "cases.json").read_text())
    report = {"model": settings.flags_model, "prompt": flags.build_prompt(), "cases": []}
    for case in cases:
        image = Image.open(folder / case["image"]).convert("RGB")
        image.thumbnail((1080, 1080))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=settings.jpeg_quality)
        started = time.monotonic()
        result = {"image": case["image"], "size": list(image.size), "expected": case["expected"]}
        try:
            detections = await flags.detect(base64.b64encode(buffer.getvalue()).decode())
            result["detections"] = detections
            counts = {}
            for item in detections:
                target = item.get("organization") or item["object_target"]
                counts[target] = counts.get(target, 0) + 1
            result["actual"] = counts
            result["passed"] = counts == case["expected"]
        except Exception as exc:
            result.update(passed=False, error=str(exc))
        result["seconds"] = round(time.monotonic() - started, 2)
        report["cases"].append(result)
        (folder / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            json.dumps({k: v for k, v in result.items() if k != "detections"}, ensure_ascii=False),
            flush=True,
        )
    passed = sum(case["passed"] for case in report["cases"])
    print(f"Passed {passed}/{len(cases)} scope checks; inspect boxes separately.", flush=True)
    return passed == len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--refresh-parameters", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.folder, args.refresh_parameters)) else 1)
