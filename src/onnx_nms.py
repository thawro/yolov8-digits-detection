import onnx_graphsurgeon as gs
import numpy as np
import onnx


def create_onnx_NMS(filepath: str = "nms.onnx", opset: int = 18):
    """Create ONNX NonMaxSupression which uses YOLO outputs to return boxes, scores and class_ids filtered out with NMS operation"""
    output0 = gs.Variable(
        name="output0", dtype=np.float32, shape=(1, "4 + num_classes", "num_boxes")
    )
    output0_transposed = gs.Variable(
        name="output0_transposed", dtype=np.float32, shape=(1, "num_boxes", "4 + num_classes")
    )
    node_transpose = gs.Node(
        op="Transpose", attrs={"perm": [0, 2, 1]}, inputs=[output0], outputs=[output0_transposed]
    )

    shape = gs.Variable(name="shape", dtype=np.int64)
    node_shape = gs.Node(
        op="Shape", attrs={"start": -1}, inputs=[output0_transposed], outputs=[shape]
    )

    boxes_starts = gs.Constant(name="boxes_starts", values=np.array([0], dtype=np.int64))
    boxes_ends = gs.Constant(name="boxes_ends", values=np.array([4], dtype=np.int64))
    boxes_axes = gs.Constant(name="boxes_axes", values=np.array([2], dtype=np.int64))
    boxes_xywh = gs.Variable(name="boxes_xywh", dtype=np.float32, shape=(1, "num_boxes", 4))
    node_slice_boxes = gs.Node(
        op="Slice",
        inputs=[output0_transposed, boxes_starts, boxes_ends, boxes_axes],
        outputs=[boxes_xywh],
    )

    scores_starts = gs.Constant(name="scores_starts", values=np.array([4], dtype=np.int64))
    scores_ends = shape
    scores_axes = gs.Constant(name="scores_axes", values=np.array([2], dtype=np.int64))
    scores = gs.Variable(name="scores", dtype=np.float32, shape=(1, "num_boxes", None))
    node_slice_scores = gs.Node(
        op="Slice",
        inputs=[output0_transposed, scores_starts, scores_ends, scores_axes],
        outputs=[scores],
    )

    class_scores = gs.Variable(name="class_scores", dtype=np.float32, shape=(1, "num_boxes", 1))
    axes = gs.Constant(name="axes", values=np.array([2], dtype=np.int64))
    node_max_scores = gs.Node(
        op="ReduceMax", attrs={"keepdims": 1}, inputs=[scores, axes], outputs=[class_scores]
    )

    int64_class_ids = gs.Variable(name="int64_class_ids", dtype=np.int64, shape=(1, "num_boxes", 1))
    node_argmax_scores = gs.Node(
        op="ArgMax", attrs={"axis": 2, "keepdims": 1}, inputs=[scores], outputs=[int64_class_ids]
    )

    class_ids = gs.Variable(name="class_ids", dtype=np.int32, shape=(1, "num_boxes", 1))
    node_cast_int64_to_int32 = gs.Node(
        op="Cast", attrs={"to": np.int32}, inputs=[int64_class_ids], outputs=[class_ids]
    )

    class_scores_transposed = gs.Variable(
        name="class_scores_transposed", dtype=np.float32, shape=(1, 1, "num_boxes")
    )
    node_class_scores_transpose = gs.Node(
        op="Transpose",
        attrs={"perm": [0, 2, 1]},
        inputs=[class_scores],
        outputs=[class_scores_transposed],
    )

    max_output_boxes_per_class = gs.Variable(
        name="max_output_boxes_per_class", dtype=np.int32, shape=(1,)
    )
    int64_max_output_boxes_per_class = gs.Variable(
        name="int64_max_output_boxes_per_class", dtype=np.int64, shape=(1,)
    )
    node_cast_int32_to_int64 = gs.Node(
        op="Cast",
        attrs={"to": np.int64},
        inputs=[max_output_boxes_per_class],
        outputs=[int64_max_output_boxes_per_class],
    )

    iou_threshold = gs.Variable(name="iou_threshold", dtype=np.float32, shape=(1,))
    score_threshold = gs.Variable(name="score_threshold", dtype=np.float32, shape=(1,))
    selected_indices = gs.Variable(
        name="selected_indices", dtype=np.int64, shape=("num_selected_boxes", 3)
    )
    node_nms = gs.Node(
        op="NonMaxSuppression",
        attrs={"center_point_box": 1},
        inputs=[
            boxes_xywh,
            class_scores_transposed,
            int64_max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        ],
        outputs=[selected_indices],
    )

    indices = gs.Constant(name="indices", values=np.array(2, dtype=np.int64))
    selected_box_indices = gs.Variable(
        name="selected_box_indices", dtype=np.int64, shape=("num_selected_boxes",)
    )
    node_selected_indices = gs.Node(
        op="Gather",
        attrs={"axis": 1},
        inputs=[selected_indices, indices],
        outputs=[selected_box_indices],
    )

    # pick boxes using NMS indices
    squeezed_boxes_xywh = gs.Variable(
        name="squeezed_boxes_xywh", dtype=np.float32, shape=("num_boxes", 4)
    )
    node_squeezed_boxes_xywh = gs.Node(
        op="Squeeze", inputs=[boxes_xywh], outputs=[squeezed_boxes_xywh]
    )

    selected_boxes_xywh = gs.Variable(
        name="selected_boxes_xywh", dtype=np.float32, shape=("num_selected_boxes", 4)
    )
    node_selected_boxes_xywh = gs.Node(
        op="Gather",
        attrs={"axis": 0},
        inputs=[squeezed_boxes_xywh, selected_box_indices],
        outputs=[selected_boxes_xywh],
    )

    # pick scores using NMS indices
    squeezed_class_scores = gs.Variable(
        name="squeezed_class_scores", dtype=np.float32, shape=("num_boxes",)
    )
    node_squeezed_class_scores = gs.Node(
        op="Squeeze", inputs=[class_scores], outputs=[squeezed_class_scores]
    )

    selected_class_scores = gs.Variable(
        name="selected_class_scores", dtype=np.float32, shape=("num_selected_boxes",)
    )
    node_selected_class_scores = gs.Node(
        op="Gather",
        attrs={"axis": 0},
        inputs=[squeezed_class_scores, selected_box_indices],
        outputs=[selected_class_scores],
    )

    # pick class_ids using NMS indices
    squeezed_class_ids = gs.Variable(
        name="squeezed_class_ids", dtype=np.int32, shape=("num_boxes",)
    )
    node_squeezed_class_ids = gs.Node(
        op="Squeeze", inputs=[class_ids], outputs=[squeezed_class_ids]
    )

    selected_class_ids = gs.Variable(
        name="selected_class_ids", dtype=np.int32, shape=("num_selected_boxes",)
    )
    node_selected_class_ids = gs.Node(
        op="Gather",
        attrs={"axis": 0},
        inputs=[squeezed_class_ids, selected_box_indices],
        outputs=[selected_class_ids],
    )

    graph = gs.Graph(
        nodes=[
            node_transpose,
            node_shape,
            node_slice_boxes,
            node_slice_scores,
            node_max_scores,
            node_argmax_scores,
            node_cast_int64_to_int32,
            node_class_scores_transpose,
            node_cast_int32_to_int64,
            node_nms,
            node_selected_indices,
            node_squeezed_boxes_xywh,
            node_selected_boxes_xywh,
            node_squeezed_class_scores,
            node_selected_class_scores,
            node_squeezed_class_ids,
            node_selected_class_ids,
        ],
        inputs=[output0, max_output_boxes_per_class, iou_threshold, score_threshold],
        outputs=[selected_boxes_xywh, selected_class_scores, selected_class_ids],
        opset=opset,
    )
    onnx.save(gs.export_onnx(graph), filepath)
