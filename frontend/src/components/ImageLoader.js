import React, { useRef } from "react";


const ImageLoader = ({ imageRef }) => {
    const inputImage = useRef(null);
    return <>
        <input
            type="file"
            ref={inputImage}
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => {
                const url = URL.createObjectURL(e.target.files[0]); // create image url
                imageRef.current.src = url; // set image source
            }}
        />
        <div className="btn-container">
            <button onClick={() => { inputImage.current.click(); }}>Open local image</button>
            {(imageRef != null) && (
                <button
                    onClick={() => {
                        inputImage.current.value = "";
                        imageRef.current.src = "#";
                    }}
                >
                    Close image
                </button>
            )}
        </div>
    </>
}

export default ImageLoader