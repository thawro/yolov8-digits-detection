import gradio as gr
import glob
import numpy as np
from src.object_detection import OnnxObjectDetector, ObjectDetector

use_onnx = True
iou_threshold = 0.7
score_threshold = 0.25

if use_onnx:
    object_detector = OnnxObjectDetector()
else:
    object_detector = ObjectDetector()


examples = glob.glob("datasets/yolo_HWD+/examples/*")


def predict(image: np.ndarray):
    if len(image.shape) == 2:
        input_image = image[..., np.newaxis]
        input_image = np.repeat(input_image, 3, axis=2)
    else:
        input_image = image
    results = object_detector(
        input_image, iou_threshold=iou_threshold, score_threshold=score_threshold
    )
    bbox_img = results.visualize(plot=False)
    return bbox_img


gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(
        source="canvas", tool="color-sketch", type="numpy", shape=(1280, 640), invert_colors=False
    ),
    outputs=gr.Image(type="numpy", shape=(1280, 640)),
    examples=examples,
).launch(server_name="0.0.0.0")
