import React, { useRef } from "react";
import { detectObjects } from "../utils/detect";
import { renderBoxes, renderInfo } from "../utils/renderCanvas";


const ImageObjectDetector = ({ session, modelInputShape, iouThreshold, scoreThreshold, maxOutputBoxesPerClass }) => {
    const imageRef = useRef(null);
    const boxesCanvasRef = useRef(null);
    const inputImage = useRef(null);
    const initCanvasHeight = modelInputShape[2]
    const initCanvasWidth = modelInputShape[3]

    const runDetection = (e) => {
        const url = URL.createObjectURL(e.target.files[0]); // create image url
        imageRef.current.src = url; // set image source
    }

    const detectAndRender = async () => {
        boxesCanvasRef.current.height = imageRef.current.height
        boxesCanvasRef.current.width = imageRef.current.width
        const { boxes, speed } = await detectObjects(
            imageRef.current,
            session,
            maxOutputBoxesPerClass,
            iouThreshold,
            scoreThreshold,
            modelInputShape
        );
        renderBoxes(imageRef, boxesCanvasRef, boxes); // Draw boxes
        renderInfo(boxesCanvasRef, speed)
    }

    return <>
        <div>
            <div>
                <input
                    type="file"
                    ref={inputImage}
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={runDetection}
                />
                <button onClick={() => { inputImage.current.click(); }}>Open local image</button>
            </div>

            <>
                <canvas id="boxesCanvas" ref={boxesCanvasRef} width={initCanvasWidth} height={initCanvasHeight} />
                <img ref={imageRef} src="#" alt="" onLoad={detectAndRender} style={{ display: "none" }} />
            </>
        </div>
    </>
};

export default ImageObjectDetector;