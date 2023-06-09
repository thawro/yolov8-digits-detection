import gradio as gr
import glob
import numpy as np
from src.onnx_model import YoloOnnxModel

model = YoloOnnxModel("models/detection_model.onnx")
examples = glob.glob("datasets/SVHN/examples/*")

iou_threshold = 0.7
conf_threshold = 0.25


def predict(gray_img):
    if len(gray_img.shape) == 2:
        img = gray_img[..., np.newaxis]
        img = np.repeat(img, 3, axis=2)
    else:
        img = gray_img
    results = model(img, iou_threshold=iou_threshold, conf_threshold=conf_threshold)
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
