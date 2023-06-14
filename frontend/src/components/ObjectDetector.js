import React, { useState, useRef, useEffect } from "react";
import { DrawableCanvas, ImageLoader, DetectionRenderer } from "./";


const ObjectDetector = ({ session, modelInputShape }) => {
    const [canvasHeight, setCanvasHeight] = useState(600)

    const [canvasWidth, setCanvasWidth] = useState(200)
    const isDrawingRef = useRef(false)
    const [isDrawing, setIsDrawing] = useState(isDrawingRef.current)

    const imageRef = useRef(null);

    const boxesCanvasRef = useRef(null);
    const sketchCanvasRef = useRef(null);

    const [lineWidth, setLineWidth] = useState(6);
    const [color, setColor] = useState('#000000');

    const handleLineWidthChange = (event) => {
        const lw = event.target.value
        sketchCanvasRef.current.getContext("2d").lineWidth = lw;
        setLineWidth(lw)
    };

    const handleColorChange = (event) => {
        const color = event.target.value
        sketchCanvasRef.current.getContext("2d").strokeStyle = color
        setColor(color)
    };

    const clearCanvas = () => {
        const sketchCtx = sketchCanvasRef.current.getContext("2d")
        const boxesCtx = boxesCanvasRef.current.getContext("2d")
        sketchCtx.fillStyle = '#FFFFFF'
        sketchCtx.fillRect(0, 0, sketchCtx.canvas.width, sketchCtx.canvas.height)
        boxesCtx.fillStyle = '#FFFFFF'
        boxesCtx.fillRect(0, 0, boxesCtx.canvas.width, boxesCtx.canvas.height)
    }



    useEffect(() => {
        const boxesCanvas = boxesCanvasRef.current
        const sketchCanvasCtx = sketchCanvasRef.current.getContext("2d")
        sketchCanvasRef.current.getContext("2d").lineWidth = lineWidth;
        boxesCanvas.addEventListener('mousedown', (event) => {
            const { offsetX, offsetY } = event
            setIsDrawing(true)
            sketchCanvasCtx.moveTo(offsetX, offsetY);
            sketchCanvasCtx.beginPath();
        });
        sketchCanvasRef.current.style.display = isDrawing ? "block" : "none"
        boxesCanvasRef.current.style.display = isDrawing ? "none" : "block"

    }, [isDrawingRef, isDrawing]);


    return <>
        <button onClick={clearCanvas}>Clear</button>
        <div>
            <DrawableCanvas
                canvasRef={sketchCanvasRef}
                isDrawing={isDrawing}
                setIsDrawing={setIsDrawing}
                isDrawingRef={isDrawingRef}
                imageRef={imageRef}
                canvasHeight={canvasHeight}
                canvasWidth={canvasWidth}
            />
            {/* <ImageLoader imageRef={imageRef} /> */}
            <DetectionRenderer
                imageRef={imageRef}
                canvasRef={boxesCanvasRef}
                session={session}
                modelInputShape={modelInputShape}
                canvasHeight={canvasHeight}
                canvasWidth={canvasWidth}
            />
        </div>
        <div>
            <label htmlFor="lineWidth">Line width: </label>
            <input id="lineWidth" type="range" min={2} max={40} step={1} value={lineWidth} onChange={handleLineWidthChange} />
        </div>
        <div>
            <label htmlFor="lineWidth">Color: </label>
            <input id="color" type="color" value={color} onChange={handleColorChange} />
        </div>
    </>
};

export default ObjectDetector;