import onnx_graphsurgeon as gs
from src.utils.utils import MODELS_PATH
from src.create_onnx_model.utils import (
    INT32,
    UINT8,
    INT64,
    FLOAT32,
    STR,
    BOOL,
    _const,
    concat,
    get_item,
    sub,
    add,
    div,
    cast,
)
import onnx


def create_onnx_postprocessing(filepath: str = "preprocessing.onnx", opset: int = 18):
    graph = gs.Graph(opset=opset)

    boxes_xywh = gs.Variable(name="boxes_xywh", dtype=FLOAT32, shape=("num_boxes", 4))
    input_h = gs.Variable(name="input_h", dtype=INT32, shape=(1,))
    input_w = gs.Variable(name="input_w", dtype=INT32, shape=(1,))
    padding_tlbr = gs.Variable(name="padding_tlbr", dtype=INT32, shape=(4,))

    inputs = [boxes_xywh, input_h, input_w, padding_tlbr]
    graph.inputs = inputs

    pad_top = graph.get_item(padding_tlbr, axis=0, idx=0, name="pad_top", shape=(1,))
    pad_left = graph.get_item(padding_tlbr, axis=0, idx=1, name="pad_left", shape=(1,))
    pad_bottom = graph.get_item(padding_tlbr, axis=0, idx=2, name="pad_bottom", shape=(1,))
    pad_right = graph.get_item(padding_tlbr, axis=0, idx=3, name="pad_right", shape=(1,))

    scalar_0 = _const("scalar_0", 0, dtype=INT32)
    boxes_subtractor_int = graph.concat(
        [pad_left, pad_top, scalar_0, scalar_0], axis=0, name="boxes_subtractor_int", shape=(4,)
    )
    boxes_subtractor = graph.cast(
        boxes_subtractor_int, dtype=FLOAT32, name="boxes_subtractor", shape=(4,)
    )
    boxes_xywh_no_pad = graph.sub(
        boxes_xywh, boxes_subtractor, name="boxes_xywh_no_pad", shape=("num_boxes", 4)
    )

    pad_y = graph.add(pad_top, pad_bottom, name="pad_y", shape=(1,))
    pad_x = graph.add(pad_left, pad_right, name="pad_x", shape=(1,))

    orig_img_h = graph.sub(input_h, pad_y, name="orig_img_h", shape=(1,))
    orig_img_w = graph.sub(input_w, pad_x, name="orig_img_w", shape=(1,))

    boxes_xywh_divider_int = graph.concat(
        [orig_img_w, orig_img_h, orig_img_w, orig_img_h],
        axis=0,
        name="boxes_xywh_divider_int",
        shape=(4,),
    )
    boxes_xywh_divider = graph.cast(
        boxes_xywh_divider_int, dtype=FLOAT32, name="boxes_xywh_divider", shape=(4,)
    )

    boxes_xywhn = graph.div(
        boxes_xywh_no_pad, boxes_xywh_divider, name="boxes_xywhn", shape=("num_boxes", 4)
    )

    graph.outputs = [boxes_xywhn]
    onnx.save(gs.export_onnx(graph), filepath)


if __name__ == "__main__":
    create_onnx_postprocessing(filepath=str(MODELS_PATH / "postprocessing.onnx"), opset=18)
