"""Run the review decoder in a killable process before publishing review media."""

import base64
import json
import math
import sys

import cv2

SAMPLE_WIDTH, SAMPLE_HEIGHT = 96, 54


def decoder_samples(path, duration):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Review decoder could not open the prepared video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25
        last_target = max(0, duration - max(1 / fps, 0.05))
        samples = []
        for fraction in (0, 0.01, 0.25, 0.5, 0.75, 0.99):
            target = min(last_target, max(0, duration * fraction))
            capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000)
            ok, frame = capture.read()
            pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not ok or frame is None or not math.isfinite(pts) or pts < 0:
                raise ValueError("Review decoder could not read a validation frame")
            pixels = cv2.resize(frame, (SAMPLE_WIDTH, SAMPLE_HEIGHT), interpolation=cv2.INTER_AREA)
            samples.append({"pts": pts, "pixels": base64.b64encode(pixels.tobytes()).decode()})
        return {"decoder_version": cv2.__version__, "samples": samples}
    finally:
        capture.release()


if __name__ == "__main__":
    print(json.dumps(decoder_samples(sys.argv[1], float(sys.argv[2]))))
