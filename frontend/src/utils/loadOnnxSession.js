import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import { download } from "../utils/download";
import { preprocessing_onnx, yolo_onnx, nms_onnx, postprocessing_onnx } from "../assets";


export const loadOnnxSession = async (setLoading, setSession, modelInputShape) => {
    cv["onRuntimeInitialized"] = async () => {
        // create session
        const preprocessingBuffer = await download(preprocessing_onnx, ["Loading Preprocessing", setLoading]);
        const preprocessing = await InferenceSession.create(preprocessingBuffer);

        const yoloBuffer = await download(yolo_onnx, ["Loading YOLOv8 model", setLoading]);
        const yolo = await InferenceSession.create(yoloBuffer);

        const nmsBuffer = await download(nms_onnx, ["Loading NMS model", setLoading]);
        const nms = await InferenceSession.create(nmsBuffer);

        const postprocessingBuffer = await download(postprocessing_onnx, ["Loading Postprocessing model", setLoading]);
        const postprocessing = await InferenceSession.create(postprocessingBuffer);

        // warmup main model
        setLoading({ text: "Warming up YOLOv8...", progress: null });
        const tensor = new Tensor("float32", new Float32Array(modelInputShape.reduce((a, b) => a * b)), modelInputShape);
        await yolo.run({ images: tensor });

        setSession({
            preprocessing: preprocessing,
            yolo: yolo,
            nms: nms,
            postprocessing: postprocessing
        });
        setLoading(null);
    };
}