import React, { useState, useRef, useEffect } from "react";
import { DrawableCanvas, DetectionRenderer } from ".";
import { CustomSlider } from "../components";


const SketchMenu = ({ lineWidth, handleLineWidthChange, color, handleColorChange, canvasWidth, handleCanvasSizeChange, canvasHeight }) => {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;

    const maxWidth = Math.floor(screenWidth * 0.8)
    const maxHeight = Math.floor(screenHeight * 0.6)

    return <div className="configMenu">
        <h3 className="configTitle">Canvas configuration</h3>
        <div className="configInputs">
            <div className="menuItem">
                <label htmlFor="lineWidth">Line width: </label>
                <CustomSlider defaultValue={lineWidth} setValue={handleLineWidthChange} min={2} max={40} step={1} />
            </div>
            <div className="menuItem">
                <label htmlFor="lineWidth">Color: </label>
                <span><input id="color" type="color" value={color} onChange={handleColorChange} /></span>
            </div>
            <div className="menuItem">
                <label htmlFor="canvasWidth">Canvas width: </label>
                <CustomSlider defaultValue={canvasWidth} setValue={(e) => handleCanvasSizeChange(e, "width")} min={100} max={maxWidth} step={10} />
            </div>
            <div className="menuItem">
                <label htmlFor="canvasHeight">Canvas height: </label>
                <CustomSlider defaultValue={canvasHeight} setValue={(e) => handleCanvasSizeChange(e, "height")} min={100} max={maxHeight} step={10} />
            </div>
        </div>
    </div>
}


const SketchObjectDetector = ({ session, modelInputShape, iouThreshold, scoreThreshold }) => {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;

    const initCanvasWidth = Math.floor(screenWidth * 0.65)
    const initCanvasHeight = Math.floor(screenHeight * 0.4)
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
        console.log("DETECTING")
        imageRef.current.src = sketchCanvasRef.current.toDataURL('image/png');
    }

    useEffect(() => {
        const sketchCanvas = sketchCanvasRef.current
        const boxesCanvas = boxesCanvasRef.current

        const sketchCtx = sketchCanvas.getContext("2d")
        sketchCtx.willReadFrequently = true

        const boxesCtx = boxesCanvas.getContext("2d")
        boxesCtx.willReadFrequently = true

        const startDrawing = () => {
            isDrawingRef.current = true
            document.documentElement.style.overflow = 'hidden';
            setIsDrawing(true)
            sketchCtx.beginPath();
        }
        const mouseStartDrawing = (e) => { startDrawing() };

        const touchStartDrawing = (e) => {
            e.preventDefault();
            startDrawing()
        };

        boxesCanvas.addEventListener('mousedown', mouseStartDrawing);
        boxesCanvas.addEventListener('touchstart', touchStartDrawing);

        sketchCanvas.style.display = isDrawing ? "block" : "none"
        boxesCanvas.style.display = isDrawing ? "none" : "block"


        return () => {
            boxesCanvas.addEventListener('mousedown', mouseStartDrawing);
            boxesCanvas.addEventListener('touchstart', touchStartDrawing);
        };

    }, [isDrawing]);

    useEffect(() => {
        runDetection()
    }, [iouThreshold, scoreThreshold])

    useEffect(() => {
        sketchCanvasRef.current.getContext("2d").lineWidth = lineWidth
        sketchCanvasRef.current.getContext("2d").strokeStyle = color
    }, [lineWidth, color])

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
        const sketchCanvas = sketchCanvasRef.current
        const sketchCtx = sketchCanvas.getContext("2d")

        const boxesCanvas = boxesCanvasRef.current
        const boxesCtx = boxesCanvas.getContext("2d")

        const size = event.target.value
        const prevSize = sketchCanvas[sizeType]
        const sketchLineWidth = sketchCtx.lineWidth
        const sketchColor = sketchCtx.strokeStyle

        const boxesData = boxesCtx.getImageData(0, 0, canvasWidth, canvasHeight);
        const sketchData = sketchCtx.getImageData(0, 0, canvasWidth, canvasHeight);

        const offset = (size - prevSize) / 2;

        sketchCanvas[sizeType] = size
        boxesCanvas[sizeType] = size
        clearCanvas()

        sketchCtx.strokeStyle = sketchColor
        sketchCtx.lineWidth = sketchLineWidth


        if (sizeType === "width") {
            boxesCtx.putImageData(boxesData, offset, 0);
            sketchCtx.putImageData(sketchData, offset, 0);
            setCanvasWidth(size)
        } else {
            boxesCtx.putImageData(boxesData, 0, offset);
            sketchCtx.putImageData(sketchData, 0, offset);
            setCanvasHeight(size)
        }
        runDetection()
    };

    const clearCanvas = () => {
        const sketchCanvas = sketchCanvasRef.current
        const sketchCtx = sketchCanvas.getContext("2d")
        const boxesCanvas = boxesCanvasRef.current
        const boxesCtx = boxesCanvas.getContext("2d")
        sketchCtx.fillStyle = '#FFFFFF'
        sketchCtx.fillRect(0, 0, sketchCanvas.width, sketchCanvas.height)
        boxesCtx.fillStyle = '#FFFFFF'
        boxesCtx.fillRect(0, 0, boxesCanvas.width, boxesCanvas.height)
        imageRef.current.src = sketchCanvas.toDataURL('image/png');
    }

    return <>
        <SketchMenu
            lineWidth={lineWidth}
            handleLineWidthChange={handleLineWidthChange}
            color={color}
            handleColorChange={handleColorChange}
            canvasWidth={canvasWidth}
            handleCanvasSizeChange={handleCanvasSizeChange}
            canvasHeight={canvasHeight}
        />
        <div>
            <DrawableCanvas
                initCanvasHeight={initCanvasHeight}
                initCanvasWidth={initCanvasWidth}
                canvasRef={sketchCanvasRef}
                setIsDrawing={setIsDrawing}
                runDetection={runDetection}
                isDrawingRef={isDrawingRef}
                canvasHeight={canvasHeight}
                canvasWidth={canvasWidth}
            />
            <DetectionRenderer
                initCanvasHeight={initCanvasHeight}
                initCanvasWidth={initCanvasWidth}
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
        <button onClick={clearCanvas}>Clear</button>
    </>
};

export default SketchObjectDetector;