import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/Loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css";
import { preprocessing_onnx, detection_onnx, nms_onnx } from "./assets";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);


  // Configs
  const modelInputShape = [1, 3, 256, 256];
  const topk = 100;
  const iouThreshold = 0.7;
  const scoreThreshold = 0.25;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    // create session
    const preprocessingBuffer = await download(preprocessing_onnx, ["Loading Preprocessing", setLoading]);
    const preprocessing = await InferenceSession.create(preprocessingBuffer);

    const modelBuffer = await download(detection_onnx, ["Loading YOLOv8 model", setLoading]);
    const model = await InferenceSession.create(modelBuffer);

    const nmsBuffer = await download(nms_onnx, ["Loading NMS model", setLoading]);
    const nms = await InferenceSession.create(nmsBuffer);

    // warmup main model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await model.run({ images: tensor });

    setSession({
      preprocessing: preprocessing,
      net: model,
      nms: nms,
    });
    setLoading(null);
  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
      <div className="header">
        <h1>Digits Detection App</h1>
        <p>
          YOLOv8 object detection application live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">YOLOv8</code> detection model trained on custom digits dataset
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            );
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />

      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button onClick={() => { inputImage.current.click(); }}>Open local image</button>
        {image && (
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;