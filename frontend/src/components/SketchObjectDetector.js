import React, { useState, useRef, useEffect } from "react";
import { DrawableCanvas, DetectionRenderer } from ".";


const SketchMenu = ({ lineWidth, handleLineWidthChange, color, handleColorChange, canvasWidth, handleCanvasSizeChange, canvasHeight }) => {
    return <div className="sketchMenu">
        <div className="menuItem">
            <label htmlFor="lineWidth">Line width: </label>
            <input id="lineWidth" type="range" min={2} max={40} step={1} value={lineWidth} onChange={handleLineWidthChange} />
        </div>
        <div className="menuItem">
            <label htmlFor="lineWidth">Color: </label>
            <input id="color" type="color" value={color} onChange={handleColorChange} />
        </div>
        <div className="menuItem">
            <label htmlFor="canvasWidth">Canvas width: </label>
            <input id="canvasWidth" type="range" min={100} max={800} step={10} value={canvasWidth}
                onChange={(e) => handleCanvasSizeChange(e, "width")}
            />
        </div>
        <div className="menuItem">
            <label htmlFor="canvasHeight">Canvas height: </label>
            <input id="canvasHeight" type="range" min={100} max={800} step={10} value={canvasHeight}
                onChange={(e) => handleCanvasSizeChange(e, "height")}
            />
        </div>
    </div>
}


const SketchObjectDetector = ({ session, modelInputShape, iouThreshold, scoreThreshold }) => {
    const initCanvasHeight = 500
    const initCanvasWidth = 500
    const [canvasHeight, setCanvasHeight] = useState(initCanvasHeight)
    const [canvasWidth, setCanvasWidth] = useState(initCanvasWidth)
    const isDrawingRef = useRef(false)
    const [isDrawing, setIsDrawing] = useState(isDrawingRef.current)

    const imageRef = useRef(null);

    const boxesCanvasRef = useRef(null);
    const sketchCanvasRef = useRef(null);

    const [lineWidth, setLineWidth] = useState(6);
    const [color, setColor] = useState('#000000');

    const runDetection = () => {
        imageRef.current.src = sketchCanvasRef.current.toDataURL('image/png');
    }

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

    const handleCanvasSizeChange = (event, sizeType) => {
        const size = event.target.value
        const sketchCanvas = sketchCanvasRef.current
        const boxesCanvas = boxesCanvasRef.current
        const sketchCanvasCtx = sketchCanvas.getContext("2d")
        const boxesCanvasCtx = boxesCanvas.getContext("2d")

        const prevSize = sketchCanvas[sizeType]

        const boxesData = boxesCanvasCtx.getImageData(0, 0, canvasWidth, canvasHeight);
        const sketchData = sketchCanvasCtx.getImageData(0, 0, canvasWidth, canvasHeight);

        const offset = (size - prevSize) / 2;

        sketchCanvas[sizeType] = size
        boxesCanvas[sizeType] = size
        clearCanvas()

        if (sizeType === "width") {
            boxesCanvasCtx.putImageData(boxesData, offset, 0);
            sketchCanvasCtx.putImageData(sketchData, offset, 0);
            setCanvasWidth(size)
        } else {
            boxesCanvasCtx.putImageData(boxesData, 0, offset);
            sketchCanvasCtx.putImageData(sketchData, 0, offset);
            setCanvasHeight(size)
        }
        runDetection()

    };

    const clearCanvas = () => {
        const sketchCanvas = sketchCanvasRef.current
        const boxesCanvas = boxesCanvasRef.current
        sketchCanvas.getContext("2d").fillStyle = '#FFFFFF'
        sketchCanvas.getContext("2d").fillRect(0, 0, sketchCanvas.width, sketchCanvas.height)
        boxesCanvas.getContext("2d").fillStyle = '#FFFFFF'
        boxesCanvas.getContext("2d").fillRect(0, 0, boxesCanvas.width, boxesCanvas.height)
        imageRef.current.src = sketchCanvasRef.current.toDataURL('image/png');
    }
    useEffect(() => {
        runDetection()
    }, [iouThreshold, scoreThreshold])

    useEffect(() => {
        boxesCanvasRef.current.height = initCanvasHeight
        boxesCanvasRef.current.width = initCanvasWidth

        sketchCanvasRef.current.height = initCanvasHeight
        sketchCanvasRef.current.width = initCanvasWidth
        clearCanvas()
    }, [])

    useEffect(() => {
        const sketchCtx = sketchCanvasRef.current.getContext("2d")

        sketchCtx.lineWidth = lineWidth

        boxesCanvasRef.current.addEventListener('pointerdown', (event) => {
            const { offsetX, offsetY } = event
            setIsDrawing(true)
            sketchCtx.moveTo(offsetX, offsetY);
            sketchCtx.beginPath();
        });
        sketchCanvasRef.current.style.display = isDrawing ? "block" : "none"
        boxesCanvasRef.current.style.display = isDrawing ? "none" : "block"

    }, [isDrawing, lineWidth]);


    return <>
        <button onClick={clearCanvas}>Clear</button>
        <div>
            <DrawableCanvas
                canvasRef={sketchCanvasRef}
                setIsDrawing={setIsDrawing}
                runDetection={runDetection}
                isDrawingRef={isDrawingRef}
                canvasHeight={canvasHeight}
                canvasWidth={canvasWidth}
            />
            <DetectionRenderer
                imageRef={imageRef}
                canvasRef={boxesCanvasRef}
                session={session}
                modelInputShape={modelInputShape}
                canvasHeight={canvasHeight}
                canvasWidth={canvasWidth}
                iouThreshold={iouThreshold}
                scoreThreshold={scoreThreshold}
            />
        </div>
        <SketchMenu
            lineWidth={lineWidth}
            handleLineWidthChange={handleLineWidthChange}
            color={color}
            handleColorChange={handleColorChange}
            canvasWidth={canvasWidth}
            handleCanvasSizeChange={handleCanvasSizeChange}
            canvasHeight={canvasHeight}
        />

    </>
};

export default SketchObjectDetector;