import numpy as np
from src.transforms import resize_pad
import onnxruntime as ort
from src.results import DetectionResults


class OnnxModel:
    def __init__(self, path):
        self.initialize_model(path)

    def initialize_model(self, path):
        self.session = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # Get model info
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        _, c, h, w = model_inputs[0].shape
        self.bbox_divider = np.array([w, h, w, h])
        self.input_w = w
        self.input_h = h
        self.input_c = c


class YoloOnnxModel(OnnxModel):
    def __init__(self, path):
        super().__init__(path)

    def __call__(self, image, iou_threshold, conf_threshold):
        return self.detect_objects(image, iou_threshold, conf_threshold)

    def prepare_input(self, image):
        image = image[..., : self.input_c]
        image, ratio, pad = resize_pad(image, self.input_w, self.input_h)
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        input_tensor = np.expand_dims(image, 0).astype(np.float32)
        return input_tensor, ratio, pad

    def detect_objects(self, image: np.ndarray, iou_threshold, conf_threshold):
        input_image = np.copy(image)
        input_tensor, ratio, pad = self.prepare_input(input_image)
        outputs = self.inference(input_tensor)
        results = self.process_output(
            input_image, outputs, ratio, pad, iou_threshold, conf_threshold
        )
        return results

    def process_output(
        self, input_image, outputs, ratio, pad, iou_threshold, conf_threshold
    ) -> DetectionResults:
        predictions = np.squeeze(outputs[0]).T

        # Filter out object confidence scores below threshold
        conf = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[conf > conf_threshold, :]
        conf = conf[conf > conf_threshold]

        if len(conf) == 0:
            return DetectionResults(orig_image=input_image)

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        pad_x, pad_y = pad
        ratio_x, ratio_y = ratio

        boxes_xywh = predictions[:, :4]
        boxes_xywh[:, 0] -= pad_x
        boxes_xywh[:, 1] -= pad_y
        w = self.input_w - pad_x * 2  # / ratio_x
        h = self.input_h - pad_y * 2  # / ratio_y
        boxes_xywh = np.divide(boxes_xywh, np.array([ratio_x, ratio_y, ratio_x, ratio_y]))
        orig_w = w / ratio_x
        orig_h = h / ratio_y
        results = DetectionResults(input_image, boxes_xywh, class_ids, conf, orig_w, orig_h)

        results.non_maximum_supression(iou_threshold)
        return results

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})
