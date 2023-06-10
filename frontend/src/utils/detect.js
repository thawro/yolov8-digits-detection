import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv8 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (
    image,
    canvas,
    session,
    topk,
    iouThreshold,
    scoreThreshold,
    inputShape
) => {
    const [modelWidth, modelHeight] = inputShape.slice(2);
    const [input, ratioX, ratioY, padLeft, padTop] = preprocessing(image, modelWidth, modelHeight);

    const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
    const config = new Tensor(
        "float32",
        new Float32Array([
            topk, // topk per class
            iouThreshold, // iou threshold
            scoreThreshold, // score threshold
        ])
    ); // nms config tensor
    const { output0 } = await session.net.run({ images: tensor }); // run session and get output layer



    const { selected } = await session.nms.run({ detection: output0, config: config }); // perform nms and filter boxes

    const boxes = [];
    // looping through output
    for (let idx = 0; idx < selected.dims[1]; idx++) {
        const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
        const box = data.slice(0, 4);
        const scores = data.slice(4); // classes probability scores
        const score = Math.max(...scores); // maximum probability scores
        const label = scores.indexOf(score); // class id of maximum probability scores

        box[0] = box[0] - padLeft
        box[1] = box[1] - padTop

        let img_w = modelWidth - padLeft * 2
        let img_h = modelHeight - padTop * 2

        let origW = img_w / ratioX
        let origH = img_h / ratioY

        let [x, y, w, h] = [box[0] / ratioX / origW, box[1] / ratioY / origH, box[2] / ratioX / origW, box[3] / ratioY / origH]

        boxes.push({
            label: label,
            probability: score,
            boundingNormalized: [x, y, w, h], // upscale box
        }); // update boxes to draw later
    }

    renderBoxes(canvas, boxes, padLeft, padTop); // Draw boxes
    input.delete(); // delete unused Mat
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */

const preprocessing = (source, W, H) => {
    const sourceImgC4 = cv.imread(source); // read from img tag
    const sourceImg = new cv.Mat(sourceImgC4.rows, sourceImgC4.cols, cv.CV_8UC3); // new image matrix
    cv.cvtColor(sourceImgC4, sourceImg, cv.COLOR_RGBA2RGB); // RGBA to BGR
    const imgH = sourceImg.rows
    const imgW = sourceImg.cols

    const aspectRatio = imgW / imgH
    let newImgH;
    let newImgW
    if (aspectRatio > 1) {
        newImgW = W
        newImgH = Math.floor(W / aspectRatio)
    } else {
        newImgH = H
        newImgW = Math.floor(H / aspectRatio)
    }
    let resizedImg = new cv.Mat(newImgH, newImgW, cv.CV_8UC3)
    let newSize = new cv.Size(newImgW, newImgH)
    cv.resize(sourceImg, resizedImg, newSize)

    let ratioX = newImgW / imgW
    let ratioY = newImgH / imgH

    let transformedImg = new cv.Mat(H, W, cv.CV_8UC3)
    let padX = W - newImgW
    let padY = H - newImgH

    let padTop = Math.floor(padY / 2)
    let padBottom = padY - padTop

    let padLeft = Math.floor(padX / 2)
    let padRight = padX - padLeft

    const fillValue = new cv.Scalar(114, 114, 114, 255)
    cv.copyMakeBorder(resizedImg, transformedImg, padTop, padBottom, padLeft, padRight, cv.BORDER_CONSTANT, fillValue)

    const input = cv.blobFromImage(
        transformedImg,
        1 / 255.0, // normalize
    ); // preprocessing image matrix
    resizedImg.delete()
    sourceImgC4.delete()
    transformedImg.delete()


    return [input, ratioX, ratioY, padLeft, padTop]
};