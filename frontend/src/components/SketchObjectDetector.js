import React, { useState, useRef, useEffect } from "react";
import { DrawableCanvas } from ".";
import { CustomSlider } from "../components";
import { detectObjects } from "../utils/detect";
import { renderBoxes, renderInfo } from "../utils/renderCanvas";


const SketchMenu = ({ lineWidth, handleLineWidthChange, color, handleColorChange, handleCanvasSizeChange, canvasWidth, canvasHeight }) => {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;

    const maxWidth = Math.floor(screenWidth * 0.8)
    const maxHeight = screenHeight

    return <div className="configMenu">
        <h3 className="configTitle">Canvas configuration</h3>
        <div className="configInputs">
            <div className="menuItem">
                <label htmlFor="lineWidth">Line width: </label>
                <CustomSlider value={lineWidth} setValue={handleLineWidthChange} min={2} max={40} step={1} />
            </div>
            <div className="menuItem">
                <label htmlFor="lineWidth">Color: </label>
                <span><input id="color" type="color" value={color} onChange={handleColorChange} /></span>
            </div>
            <div className="menuItem">
                <label htmlFor="canvasWidth">Canvas width: </label>
                <CustomSlider value={canvasWidth} setValue={(e) => handleCanvasSizeChange(e, "width")} min={100} max={maxWidth} step={10} />
            </div>
            <div className="menuItem">
                <label htmlFor="canvasHeight">Canvas height: </label>
                <CustomSlider value={canvasHeight} setValue={(e) => handleCanvasSizeChange(e, "height")} min={100} max={maxHeight} step={10} />
            </div>
        </div>
    </div>
}


const SketchObjectDetector = ({ session, modelInputShape, maxOutputBoxesPerClass, iouThreshold, scoreThreshold }) => {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;

    const initCanvasWidth = Math.floor(screenWidth * 0.65)
    const initCanvasHeight = Math.floor(screenHeight * 0.4)
    const [canvasHeight, setCanvasHeight] = useState(initCanvasHeight)
    const [canvasWidth, setCanvasWidth] = useState(initCanvasWidth)
    const isDrawingRef = useRef(false)
    const [isDrawing, setIsDrawing] = useState(isDrawingRef.current)

    const imageRef = useRef(null);
    const inputImageRef = useRef(null);
    const localImageRef = useRef(null);


    const boxesCanvasRef = useRef(null);
    const sketchCanvasRef = useRef(null);

    const [lineWidth, setLineWidth] = useState(6);
    const [color, setColor] = useState('#000000');


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

    const updateCanvasProps = ({ height, width, lineWidth, strokeStyle }) => {
        const sketchCanvas = sketchCanvasRef.current
        const sketchCtx = sketchCanvas.getContext("2d")

        const boxesCanvas = boxesCanvasRef.current
        sketchCanvas.width = width
        boxesCanvas.width = width

        sketchCanvas.height = height
        boxesCanvas.height = height
        sketchCtx.strokeStyle = strokeStyle
        sketchCtx.lineWidth = lineWidth

        setCanvasWidth(width)
        setCanvasHeight(height)

    }

    const changeCanvasSize = ({ w, h }) => {
        const sketchCanvas = sketchCanvasRef.current
        const sketchCtx = sketchCanvas.getContext("2d")

        const boxesCanvas = boxesCanvasRef.current
        const boxesCtx = boxesCanvas.getContext("2d")

        const prevWidth = sketchCanvas.width
        const prevHeight = sketchCanvas.height

        const boxesData = boxesCtx.getImageData(0, 0, canvasWidth, canvasHeight);
        const sketchData = sketchCtx.getImageData(0, 0, canvasWidth, canvasHeight);

        const offsetX = (w - prevWidth) / 2;
        const offsetY = (h - prevHeight) / 2;

        updateCanvasProps({ width: w, height: h, lineWidth: lineWidth, strokeStyle: color })
        clearCanvas()

        boxesCtx.putImageData(boxesData, offsetX, offsetY);
        sketchCtx.putImageData(sketchData, offsetX, offsetY);

        runDetection()
    }

    const handleCanvasSizeChange = (event, sizeType) => {
        const sketchCanvas = sketchCanvasRef.current
        const size = event.target.value
        const params = sizeType === "width" ? { w: size, h: sketchCanvas.height } : { w: sketchCanvas.width, h: size }
        changeCanvasSize({ ...params })
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


    const runDetection = () => {
        imageRef.current.src = sketchCanvasRef.current.toDataURL('image/png');
    }

    const detectAndRender = async () => {
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


    const loadImage = (e) => {
        const url = URL.createObjectURL(e.target.files[0]); // create image url
        localImageRef.current.src = url; // set image source
    }

    const putLocalImageOnCanvas = async () => {
        const boxesCanvas = boxesCanvasRef.current
        const sketchCanvas = sketchCanvasRef.current

        const boxesCtx = boxesCanvas.getContext("2d")
        const sketchCtx = sketchCanvas.getContext("2d")

        updateCanvasProps(
            {
                width: localImageRef.current.width,
                height: localImageRef.current.height,
                lineWidth: lineWidth,
                strokeStyle: color
            }
        )
        boxesCtx.drawImage(localImageRef.current, 0, 0)
        sketchCtx.drawImage(localImageRef.current, 0, 0)
        runDetection()
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
            <input type="file" ref={inputImageRef} accept="image/*" style={{ display: "none" }} onChange={loadImage} />
            <button onClick={() => { inputImageRef.current.click(); }}>Open local image</button>
            <button onClick={clearCanvas}>Clear canvas</button>
            <img ref={localImageRef} src="#" alt="" onLoad={putLocalImageOnCanvas} style={{ display: "none" }} />
        </div>
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
            <>
                <canvas id="boxesCanvas" ref={boxesCanvasRef} width={initCanvasWidth} height={initCanvasHeight} />
                <img ref={imageRef} src="#" alt="" onLoad={detectAndRender} style={{ display: "none" }} />
            </>
        </div>

    </>
};

export default SketchObjectDetector;