import React, { useState, useRef, useEffect } from "react";
import { DrawableCanvas } from ".";
import { CustomSlider } from "../components";
import { detectObjects } from "../utils/detect";
import { renderBoxes, renderInfo } from "../utils/renderCanvas";
import { exampleImages, exampleVideos } from "../constants"
import { saveAs } from 'file-saver';
import { BsPlayCircleFill, BsPauseCircleFill } from "react-icons/bs";

const isVideoPlaying = video => !video.paused && !video.ended


const SketchConfigMenu = ({ lineWidth, handleLineWidthChange, color, handleColorChange, handleCanvasSizeChange, canvasWidth, canvasHeight }) => {
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

const ImageExample = ({ src, loadImage }) => {
    const size = 75
    const handleLoadImage = (e) => {
        loadImage(src)
    }
    return <img className="exampleImage" width={size} height={size} src={src} onClick={handleLoadImage} />
}

const VideoExample = ({ ref, src, handleVideoClick, style }) => {
    const size = 75
    const playerRef = useRef(ref)

    const handleClick = (e) => {
        handleVideoClick(playerRef)
    }

    return <div className="exampleVideo" onClick={handleClick}>
        <video ref={playerRef} src={src} width={size} height={size} style={style} />
        <BsPlayCircleFill className="playButton" />
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

    const uploadedVideoRef = useRef(null)

    const imageRef = useRef(null);
    const uploadFileRef = useRef(null);
    const localImageRef = useRef(null);

    const videoRef = useRef(null);
    const idRef = useRef(null)

    const mediaType = useRef("image")

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
            // document.documentElement.style.overflow = 'hidden'; // TODO
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
        if (sketchCanvas.width === width && sketchCanvas.height === height) {
            return // no need to update
        }
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

    const uploadFile = (file) => {
        const src = URL.createObjectURL(file)
        if (file.type.includes("image")) {
            loadImage(src);
        } else if (file.type.includes("video")) {
            loadVideo(src)
        }
    }


    const loadImage = (src) => {
        mediaType.current = "image"
        if (videoRef.current != null) {
            if (!videoRef.current.paused) {
                videoRef.current.pause()
            }
        }
        localImageRef.current.src = src; // set image source
    }

    const loadVideo = (src) => {
        mediaType.current = "video"
        // if (uploadedVideoRef.current.src !== "") {
        //     console.log("pausing previous video by id")
        //     exampleVideoRef.current.pause()
        // }
        videoRef.current.pause()
        clearTimeout(idRef.current);
        uploadedVideoRef.current.src = src
        console.log("Loading new video")
        uploadedVideoRef.current.load()
        uploadedVideoRef.current.style.visibility = "visible"

        // uploadedVideoRef.current.addEventListener("loadedmetadata", function (e) {
        //     updateCanvasProps({ width: exampleVideoRef.current.videoWidth, height: exampleVideoRef.current.videoHeight, lineWidth: lineWidth, strokeStyle: color })
        // })
        uploadedVideoRef.current.addEventListener("loadeddata", function (e) {
            console.log("LOADED")

            if (!isVideoPlaying(uploadedVideoRef.current)) {
                console.log("PLAYED")
                playVideo(uploadedVideoRef)
            }
        })
    }

    const putVideoOnCanvas = (playerRef) => {
        if (mediaType.current !== "video") { return }
        if (playerRef.current.src !== videoRef.current.src) { return }
        console.log("Video on canvas ", playerRef.current.src)
        const sketchCanvas = sketchCanvasRef.current
        const sketchCtx = sketchCanvas.getContext("2d")
        sketchCtx.drawImage(playerRef.current, 0, 0, sketchCanvas.width, sketchCanvas.height)
        runDetection()
        if (!playerRef.current.paused && !playerRef.current.ended) {
            idRef.current = setTimeout(putVideoOnCanvas, 1000 / 15, playerRef)
        }
    }

    const playVideo = (playerRef) => {
        const height = playerRef.current.videoHeight
        const width = playerRef.current.videoWidth
        videoRef.current = playerRef.current
        updateCanvasProps({ width: width, height: height, lineWidth: lineWidth, strokeStyle: color })
        playerRef.current.play()
        putVideoOnCanvas(playerRef)
    }

    const handleVideoClick = (playerRef) => {
        mediaType.current = "video"
        if (videoRef.current.src === "") { // first time
            console.log("FIRST TIME")
            playVideo(playerRef)
        } else {
            if (playerRef.current.src === videoRef.current.src) { // clicked the same video
                if (playerRef.current.paused) {
                    console.log("PLAY SAME")
                    playVideo(playerRef)
                } else {
                    console.log("PAUSE SAME")
                    playerRef.current.pause()
                    clearTimeout(idRef.current);
                }
            } else { // clicked other video
                console.log("PLAY NEW")
                videoRef.current.pause()
                clearTimeout(idRef.current);
                playVideo(playerRef)
            }
        }
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



    const saveCanvas = () => {
        boxesCanvasRef.current.toBlob(function (blob) {
            saveAs(blob, "predictions.png");
        });
    }


    return <>
        <div className="sketchMenu">
            <SketchConfigMenu
                lineWidth={lineWidth}
                handleLineWidthChange={handleLineWidthChange}
                color={color}
                handleColorChange={handleColorChange}
                canvasWidth={canvasWidth}
                handleCanvasSizeChange={handleCanvasSizeChange}
                canvasHeight={canvasHeight}
            />
            <div>
                <input
                    type="file" ref={uploadFileRef}
                    accept="image/*,video/*" style={{ display: "none" }}
                    onChange={(e) => uploadFile(e.target.files[0])} />
                <button onClick={() => { uploadFileRef.current.click(); }}>Upload</button>
                <button onClick={clearCanvas}>Clear canvas</button>
                <button onClick={saveCanvas}>Save</button>
            </div>
            <div>
                <p>Examples: </p>
                <div className="examples">
                    {exampleImages.map((example, index) => (
                        <ImageExample key={index} src={example} loadImage={loadImage} />
                    ))}
                    {exampleVideos.map((example, index) => (
                        <VideoExample src={example} handleVideoClick={handleVideoClick} />
                    ))}
                    <div className="exampleVideo" onClick={(e) => { handleVideoClick(uploadedVideoRef) }}>
                        <video
                            id={"uploadedVideo"}
                            ref={uploadedVideoRef}
                            width={75}
                            height={75}
                            style={{ visibility: "hidden" }}
                        />
                        <BsPlayCircleFill className="playButton" />
                    </div>
                    <video
                        id={"currentlyPlayedVideo"}
                        ref={videoRef}
                        width={75}
                        height={75}
                        style={{ visibility: "hidden" }}
                    />

                </div>
            </div>
        </div>
        <div className="sketchField">
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
                <img id="modelInput" ref={imageRef} src="#" alt="" onLoad={detectAndRender} style={{ visibility: "hidden", display: "none" }} />
            </>
        </div>
        <img id="upladedImage" ref={localImageRef} src="#" alt="" onLoad={putLocalImageOnCanvas} style={{ visibility: "hidden", display: "none" }} />

    </>
};

export default SketchObjectDetector;