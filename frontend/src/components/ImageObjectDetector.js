import React, { useState, useRef, useEffect } from "react";
import { ImageLoader, DetectionRenderer } from ".";


const ImageObjectDetector = ({ session, modelInputShape }) => {
    const imageRef = useRef(null);
    const boxesCanvasRef = useRef(null);
    const canvasHeight = modelInputShape[2]
    const canvasWidth = modelInputShape[3]

    return <>
        <div>
            <ImageLoader imageRef={imageRef} />
            <DetectionRenderer
                imageRef={imageRef}
                canvasRef={boxesCanvasRef}
                session={session}
                modelInputShape={modelInputShape}
                canvasHeight={canvasHeight}
                canvasWidth={canvasWidth}
            />
        </div>
    </>
};

export default ImageObjectDetector;