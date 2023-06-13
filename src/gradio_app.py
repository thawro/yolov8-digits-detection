import gradio as gr
import glob
import numpy as np
from src.object_detection import OnnxObjectDetector

object_detector = OnnxObjectDetector(
    preprocessing_path="models/preprocessing.onnx",
    yolo_path="models/detection_model.onnx",
    nms_path="models/nms.onnx",
)


examples = glob.glob("datasets/SVHN/examples/*")

iou_threshold = 0.7
score_threshold = 0.25


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
