import React from "react";
import { detectObjects } from "../utils/detect";
import { renderBoxes, renderInfo } from "../utils/renderCanvas";



const DetectionRenderer = ({ initCanvasHeight, initCanvasWidth, imageRef, canvasRef, session, modelInputShape, iouThreshold, scoreThreshold }) => {
    const max_output_boxes_per_class = 100;

    const detectAndRender = async () => {
        const { boxes, speed } = await detectObjects(
            imageRef.current,
            session,
            max_output_boxes_per_class,
            iouThreshold,
            scoreThreshold,
            modelInputShape
        );
        renderBoxes(imageRef, canvasRef, boxes); // Draw boxes
        renderInfo(canvasRef, speed)
    }

    return <>
        <canvas
            id="canvas"
            ref={canvasRef}
            width={initCanvasWidth}
            height={initCanvasHeight}

        />
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