import React, { useEffect } from "react";
import "../style/loader.css";

function getMousePosition(e, canvas) {
    var mouseX = e.offsetX * canvas.width / canvas.clientWidth | 0;
    var mouseY = e.offsetY * canvas.height / canvas.clientHeight | 0;
    return { x: mouseX, y: mouseY };
}


const DrawableCanvas = ({ initCanvasHeight, initCanvasWidth, canvasRef, runDetection, setIsDrawing, isDrawingRef }) => {
    let canvas, ctx;

    useEffect(() => {
        canvas = canvasRef.current
        ctx = canvas.getContext('2d')
        ctx.willReadFrequently = true

        ctx.fillStyle = '#FFFFFF'; // White color
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const draw = (x, y) => {
            if (!isDrawingRef.current) { return }
            ctx.lineTo(x, y);
            ctx.stroke();
        }

        const stopDrawing = () => {
            runDetection()
            isDrawingRef.current = false
            // document.documentElement.style.overflow = 'auto'; //TODO
            setIsDrawing(false)
        }

        const touchDraw = (e) => {
            e.preventDefault()
            e.stopPropagation();
            const canvasRect = canvas.getBoundingClientRect()
            const scrollTop = document.documentElement.scrollTop
            const { pageX, pageY } = e.touches[0]
            const x = pageX - canvasRect.x
            const y = pageY - canvasRect.y - scrollTop
            draw(x, y)
        };

        const mouseDraw = (e) => {
            const { x, y } = getMousePosition(e, canvas)
            draw(x, y)
        };

        const mouseStopDrawing = (e) => { stopDrawing() };

        const touchStopDrawing = (e) => {
            e.preventDefault();
            stopDrawing()
        };

        canvas.addEventListener('mousemove', mouseDraw);
        canvas.addEventListener('touchmove', touchDraw);
        canvas.addEventListener('mouseup', mouseStopDrawing);
        canvas.addEventListener('touchend', touchStopDrawing);
        return () => {
            canvas.addEventListener('mousemove', mouseDraw);
            canvas.addEventListener('touchmove', touchDraw);
            canvas.addEventListener('mouseup', mouseStopDrawing);
            canvas.addEventListener('touchend', touchStopDrawing);
        };
    }, []);

    return <>
        <canvas
            ref={canvasRef}
            id="sketchCanvas"
            width={initCanvasWidth} height={initCanvasHeight}
        />
    </>
}

export default DrawableCanvas;