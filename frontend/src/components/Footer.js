import React from "react";



const Footer = () => {
    return <div className="footer">
        <p>
            Serving : <code className="code">YOLOv8</code> detection model trained on custom digits dataset
        </p>
        <p>
            All parts of the pipeline are done with ONNX models
            <br />
            (preprocessing, object detection, non maximum supression and postprocessing)
        </p>
    </div>
}

export default Footer