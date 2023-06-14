import React from "react";



const Header = () => {
    return <div className="header">
        <h1>Digits Detection App</h1>
        <p>
            Object detection pipeline powered by &nbsp; <code className="code">onnxruntime-web</code>
        </p>
        <p>&nbsp;</p>
        <p>
            Instruction: draw digits on the white canvas below. The model detects digits after each line drawn.
        </p>
    </div>
}

export default Header