import React, { useEffect } from "react";
import "../style/loader.css";


const DrawableCanvas = ({ canvasRef, runDetection, setIsDrawing, isDrawingRef }) => {

    useEffect(() => {
        const canvas = canvasRef.current
        const context = canvas.getContext('2d')
        runDetection() // predict on whiteboard to show it as output

        context.fillStyle = '#FFFFFF'; // White color
        context.fillRect(0, 0, canvas.width, canvas.height);

        const draw = (event) => {

            const { offsetX, offsetY } = event;
            context.lineTo(offsetX, offsetY);
            context.stroke();
        };

        const stopDrawing = () => {
            isDrawingRef.current = false
            setIsDrawing(false)
            runDetection()
        };

        canvas.addEventListener('pointermove', draw);
        canvas.addEventListener('pointerup', stopDrawing);
        canvas.addEventListener('pointerout', stopDrawing);

        return () => {
            canvas.addEventListener('pointermove', draw);
            canvas.addEventListener('pointerup', stopDrawing);
            canvas.addEventListener('pointerout', stopDrawing);
        };
    }, [canvasRef, isDrawingRef, setIsDrawing]);


    return <>
        <canvas
            ref={canvasRef}
            id="sketchCanvas"
        />


    </>
}

export default DrawableCanvas;