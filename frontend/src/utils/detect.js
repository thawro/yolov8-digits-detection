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
    const sourceImgC4 = cv.imread(image); // read from img tag
    const preprocessed = preprocessing(sourceImgC4, modelWidth, modelHeight);
    const input = preprocessed.input
    const [padLeft, padRight, padTop, padBottom] = preprocessed.pad
    const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
    const { output0 } = await session.net.run({ images: tensor }); // run session and get output layer
    const { selected_boxes_xywh, selected_class_scores, selected_class_ids } = await session.nms.run(
        {
            output0: output0,
            int32_max_output_boxes_per_class: new Tensor("int32", new Int32Array([topk])),
            iou_threshold: new Tensor("float32", new Float32Array([iouThreshold])),
            score_threshold: new Tensor("float32", new Float32Array([scoreThreshold])),
        }
    )

    const boxes = []
    const numBoxes = selected_boxes_xywh.dims[0]
    for (let idx = 0; idx < numBoxes; idx++) {
        const boxStartIdx = idx * 4
        const box_xywh = selected_boxes_xywh.data.subarray(boxStartIdx, boxStartIdx + 4)
        const label = selected_class_ids.data[idx]
        const conf = selected_class_scores.data[idx]

        box_xywh[0] -= padLeft
        box_xywh[1] -= padTop
        const w = modelWidth - (padLeft + padRight)
        const h = modelHeight - (padTop + padBottom)

        // normalize
        box_xywh[0] /= w
        box_xywh[1] /= h
        box_xywh[2] /= w
        box_xywh[3] /= h

        boxes.push({
            label: label,
            conf: conf,
            box_xywhn: box_xywh,
        })
    }
    renderBoxes(canvas, boxes); // Draw boxes
    input.delete(); // delete unused Mat
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */

const preprocessing = (sourceImgC4, W, H) => {
    // const sourceImgC4 = cv.imread(source); // read from img tag
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


    return {
        input: input,
        pad: [padLeft, padRight, padTop, padBottom]
    }
};