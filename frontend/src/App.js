import React, { useState } from "react";
import { Loader, Header, SketchObjectDetector, ImageObjectDetector, Footer } from "./components";
import "./style/App.css";
import { loadOnnxSession } from "./utils/loadOnnxSession";


const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const modelInputShape = [1, 3, 256, 256];
  const [iouThreshold, setIouThreshold] = useState(0.7)
  const [scoreThreshold, setScoreThreshold] = useState(0.25)

  loadOnnxSession(setLoading, setSession, modelInputShape)

  const detectorProps = {
    session: session,
    modelInputShape: modelInputShape,
    iouThreshold: iouThreshold,
    scoreThreshold: scoreThreshold
  }

  return (
    <div className="App" style={{ height: loading ? "100vh" : "100%" }}>

      {loading ?
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
        :

        <>
          <Header />

          <div className="detector">
            <div className="menuItem">
              <label htmlFor="scoreThreshold">Confidence threshold: </label>
              <input id="scoreThreshold" type="range" min={0} max={1} step={0.01} value={scoreThreshold} onChange={(e) => setScoreThreshold(e.target.value)} />
            </div>
            <div className="menuItem">
              <label htmlFor="iouThreshold">IoU threshold: </label>
              <input id="iouThreshold" type="range" min={0} max={1} step={0.01} value={iouThreshold} onChange={(e) => setIouThreshold(e.target.value)} />
            </div>
            <SketchObjectDetector {...detectorProps} />
          </div>
          <Footer />
        </>
      }

    </div>
  );
};

export default App;