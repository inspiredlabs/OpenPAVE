"""Execute the bundled BlazePalm model and pass its raw tensors to our decoder."""
from __future__ import annotations

import json
import platform
from pathlib import Path
import zipfile

import numpy as np
import tensorflow as tf

from .palm_decoder import decode


def main() -> None:
    task = Path(__file__).resolve().parents[1] / "weights" / "hand_landmarker.task"
    with zipfile.ZipFile(task) as archive:
        model = archive.read("hand_detector.tflite")
    interpreter = tf.lite.Interpreter(model_content=model, num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"],
                           np.zeros(input_detail["shape"], dtype=input_detail["dtype"]))
    interpreter.invoke()
    outputs = [interpreter.get_tensor(item["index"])
               for item in interpreter.get_output_details()]
    regressors = next(value for value in outputs if value.shape[-1] == 18)
    scores = next(value for value in outputs if value.shape[-1] == 1)
    print(json.dumps({
        "tensorflow": tf.__version__, "machine": platform.machine(),
        "input": [int(value) for value in input_detail["shape"]],
        "outputs": [[int(axis) for axis in value.shape] for value in outputs],
        "zero_frame_detections": len(decode(regressors, scores)),
        "score_range": [float(scores.min()), float(scores.max())],
    }, indent=2))


if __name__ == "__main__":
    main()
