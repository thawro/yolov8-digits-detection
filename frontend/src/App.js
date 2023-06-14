import React, { useState } from "react";
import { Loader, Header, ObjectDetector, Footer } from "./components";
import "./style/App.css";
import { loadOnnxSession } from "./utils/loadOnnxSession";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const modelInputShape = [1, 3, 256, 256];

  loadOnnxSession(setLoading, setSession, modelInputShape)

  return (
    <div className="App">
      <Header />
      {loading ? (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      ) : (
        <>

          <ObjectDetector session={session} modelInputShape={modelInputShape} />

        </>

      )}
      <Footer />
      <canvas id="canv" />
    </div>
  );
};

export default App;