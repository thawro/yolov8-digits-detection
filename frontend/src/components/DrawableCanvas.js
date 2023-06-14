import React, { useRef, useEffect, useState } from "react";
import "../style/loader.css";


const DrawableCanvas = ({ canvasRef, isDrawing, setIsDrawing, isDrawingRef, imageRef, canvasHeight, canvasWidth }) => {

    useEffect(() => {
        const canvas = canvasRef.current
        const context = canvas.getContext('2d')
        imageRef.current.src = canvas.toDataURL('image/png');

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
            imageRef.current.src = canvas.toDataURL('image/png');
        };


        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        return () => {
            canvas.removeEventListener('mousemove', draw);
            canvas.removeEventListener('mouseup', stopDrawing);
            canvas.removeEventListener('mouseout', stopDrawing);
        };
    }, []);


    return <>
        <canvas
            ref={canvasRef}
            id="sketchCanvas"
            width={canvasWidth}
            height={canvasHeight}
        />


    </>
}

export default DrawableCanvas;