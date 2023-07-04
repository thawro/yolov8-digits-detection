# **About**
Handwritten digits detection using a [YOLOv8](https://docs.ultralytics.com/modes/) detection model and ONNX pre/post processing.
An example of how model works in real world scenario can be viewed at **[https://thawro.github.io/web-object-detector/](https://thawro.github.io/web-object-detector/)**.

## Data
The dataset consists of images created with the use of a [HWD+](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9702948/) dataset (more [here](https://github.com/thawro/yolov8-digits-detection#data)).

## Pipeline
Each pipeline step is done with ONNX models. The complete pipeline during inference is the following:
1. Image preprocessing - resize and pad to match model input size ([preprocessing](models/preprocessing.onnx))
2. Object detection - Detect objects with YOLOv8 model ([yolo](models/yolo.onnx))
3. Non Maximum Supression - Apply NMS to YOLO output ([nms](models/nms.onnx))
4. Postprocessing - Apply postprocessing to filtered boxes ([postprocessing](models/postprocessing.onnx))


# **Tech stack**
* [PyTorch](https://pytorch.org/) - neural networks architectures and datasets classes
* [ONNX](https://onnx.ai/) - All processing steps used in [pipeline](#pipeline)
* [ONNX Runtime](https://onnxruntime.ai/) - Pipeline inference
* [OpenCV](https://opencv.org/) - Image processing for the server-side model inference (optional)
* [React](https://react.dev/) - Web application used to test object detection models in real world examples


# **App instruction**
1. Go to **[https://thawro.github.io/web-object-detector/](https://thawro.github.io/web-object-detector/)**
2. Follow the instructions on the page
