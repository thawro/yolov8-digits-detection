import React from "react";
import { detectObjects } from "../utils/detect";
import { renderBoxes } from "../utils/renderBox";



const DetectionRenderer = ({ imageRef, canvasRef, session, modelInputShape, canvasHeight, canvasWidth }) => {
    const max_output_boxes_per_class = 100;
    const iouThreshold = 0.7;
    const scoreThreshold = 0.25;

    const detectAndRender = async () => {
        const boxes = await detectObjects(
            imageRef.current,
            session,
            max_output_boxes_per_class,
            iouThreshold,
            scoreThreshold,
            modelInputShape
        );
        renderBoxes(imageRef, canvasRef, boxes); // Draw boxes
    }

    return <>
        <canvas id="canvas" height={canvasHeight} width={canvasWidth} ref={canvasRef} />
        <img
            ref={imageRef}
            src="#"
            alt=""
            onLoad={detectAndRender}
            style={{ display: "none" }}
        />
    </>
}

export default DetectionRenderer