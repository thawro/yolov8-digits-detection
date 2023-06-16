import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";

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
export const detectObjects = async (
    image,
    session,
    max_output_boxes_per_class,
    iouThreshold,
    scoreThreshold,
    inputShape
) => {
    const [modelHeight, modelWidth] = inputShape.slice(2);
    const sourceImgC4 = cv.imread(image); // read from img tag

    const useONNX = true

    const model_input_h = new Tensor("int32", new Int32Array([modelHeight]))
    const model_input_w = new Tensor("int32", new Int32Array([modelWidth]))
    const imageShape = [sourceImgC4.rows, sourceImgC4.cols, sourceImgC4.channels()]

    let startTime = new Date();
    let preprocessed
    if (useONNX) {
        const inp_image = new Tensor("uint8", sourceImgC4.data, imageShape); // to ort.Tensor
        preprocessed = await session.preprocessing.run(
            {
                image: inp_image,
                input_h: model_input_h,
                input_w: model_input_w,
                fill_value: new Tensor("uint8", new Uint8Array([114])),
            }
        ) // preprocess image: resize, pad and normalize
    } else {
        preprocessed = preprocessing(sourceImgC4, modelHeight, modelWidth);
        preprocessed.preprocessed_img = new Tensor("float32", new Float32Array(preprocessed.preprocessed_img.data32F), inputShape);
        preprocessed.padding_tlbr = new Tensor("int32", new Int32Array(preprocessed.padding_tlbr), [4]);
    }
    const { preprocessed_img, padding_tlbr } = preprocessed
    const preprocessingTime = new Date() - startTime;


    startTime = new Date();
    const tensor_img = new Tensor("float32", preprocessed_img.data, inputShape);
    const { output0 } = await session.yolo.run({ images: tensor_img }); // run yolo on preprocessed image and get outputs
    const detectionTime = new Date() - startTime;


    startTime = new Date();
    const { selected_boxes_xywh, selected_class_scores, selected_class_ids } = await session.nms.run(
        {
            output0: output0,
            max_output_boxes_per_class: new Tensor("int32", new Int32Array([max_output_boxes_per_class])),
            iou_threshold: new Tensor("float32", new Float32Array([iouThreshold])),
            score_threshold: new Tensor("float32", new Float32Array([scoreThreshold])),
        }
    ) // filter out boxes with Non Max Supression
    const nmsTime = new Date() - startTime;


    startTime = new Date();
    const numBoxes = selected_boxes_xywh.dims[0]
    let boxes_xywhn_2d
    if (useONNX) {
        const { boxes_xywhn } = await session.postprocessing.run(
            {
                input_h: model_input_h,
                input_w: model_input_w,
                boxes_xywh: selected_boxes_xywh,
                padding_tlbr: padding_tlbr
            }
        ); // postprocess boxes (normalize wrt to the original img size)
        boxes_xywhn_2d = []
        for (let idx = 0; idx < numBoxes; idx++) {
            const boxStartIdx = idx * 4
            const box = boxes_xywhn.data.subarray(boxStartIdx, boxStartIdx + 4)
            boxes_xywhn_2d.push(box)
        }
    } else {
        boxes_xywhn_2d = postprocessing(modelHeight, modelWidth, selected_boxes_xywh, padding_tlbr).boxes_xywhn
    }
    const postprocessingTime = new Date() - startTime;

    const boxes = []
    for (let idx = 0; idx < numBoxes; idx++) {
        boxes.push({
            label: selected_class_ids.data[idx],
            conf: selected_class_scores.data[idx],
            box_xywhn: boxes_xywhn_2d[idx],
        })
    }
    sourceImgC4.delete()
    const speed = {
        preprocessing: preprocessingTime,
        detection: detectionTime,
        nms: nmsTime,
        postprocessing: postprocessingTime
    }
    return { boxes: boxes, speed: speed }
};

/**
 * Preprocessing image
 * @param {cv.Mat} sourceImgC4 image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */

const preprocessing = (sourceImgC4, modelHeight, modelWidth) => {
    // const sourceImgC4 = cv.imread(source); // read from img tag
    const sourceImg = new cv.Mat(sourceImgC4.rows, sourceImgC4.cols, cv.CV_8UC3); // new image matrix
    cv.cvtColor(sourceImgC4, sourceImg, cv.COLOR_RGBA2RGB); // RGBA to BGR
    const imgH = sourceImg.rows
    const imgW = sourceImg.cols

    const aspectRatio = imgW / imgH
    let newImgH;
    let newImgW
    if (aspectRatio > 1) {
        newImgW = modelWidth
        newImgH = Math.floor(modelWidth / aspectRatio)
    } else {
        newImgH = modelHeight
        newImgW = Math.floor(modelHeight * aspectRatio)
    }
    let resizedImg = new cv.Mat(newImgH, newImgW, cv.CV_8UC3)
    let newSize = new cv.Size(newImgW, newImgH)
    cv.resize(sourceImg, resizedImg, newSize)

    let transformedImg = new cv.Mat(modelHeight, modelWidth, cv.CV_8UC3)
    let padX = modelWidth - newImgW
    let padY = modelHeight - newImgH

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
    transformedImg.delete()
    return {
        preprocessed_img: input,
        padding_tlbr: [padTop, padLeft, padBottom, padRight]
    }
};


/**
 * Preprocessing image
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @param {Tensor} boxesXYWH model input height
 * @param {Tensor} paddingTLBR model input height
 * @return preprocessed image and configs
 */

const postprocessing = (modelHeight, modelWidth, boxesXYWH, paddingTLBR) => {
    const [padTop, padLeft, padBottom, padRight] = paddingTLBR.data
    const boxesXYWHN = []
    const numBoxes = boxesXYWH.dims[0]
    for (let idx = 0; idx < numBoxes; idx++) {
        const boxStartIdx = idx * 4
        const box_xywh = boxesXYWH.data.subarray(boxStartIdx, boxStartIdx + 4)

        box_xywh[0] -= padLeft
        box_xywh[1] -= padTop
        const w = modelWidth - (padLeft + padRight)
        const h = modelHeight - (padTop + padBottom)

        // normalize
        box_xywh[0] /= w
        box_xywh[1] /= h
        box_xywh[2] /= w
        box_xywh[3] /= h

        boxesXYWHN.push(box_xywh)
    }
    return { boxes_xywhn: boxesXYWHN }
};