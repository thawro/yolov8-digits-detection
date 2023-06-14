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
    cast,
    slice_tensor,
    get_shape,
    get_item,
    if_statement,
    div,
    identity,
    resize,
    sub,
    concat,
    pad_image,
    transpose,
    mul,
)
import onnx


def create_onnx_preprocessing(filepath: str = "preprocessing.onnx", opset: int = 18):
    graph = gs.Graph(opset=opset)

    input_h = gs.Variable(name="input_h", dtype=INT32, shape=(1,))
    input_w = gs.Variable(name="input_w", dtype=INT32, shape=(1,))
    image = gs.Variable(name="image", dtype=UINT8, shape=("height", "width", "channels"))
    fill_value = gs.Variable(name="fill_value", dtype=UINT8, shape=(1,))

    inputs = [image, input_h, input_w, fill_value]
    graph.inputs = inputs

    input_h_int64 = graph.cast(input_h, dtype=INT64, name="input_h_int64", shape=(1,))
    input_w_int64 = graph.cast(input_w, dtype=INT64, name="input_w_int64", shape=(1,))

    input_h_float = graph.cast(input_h, dtype=FLOAT32, name="input_h_float", shape=(1,))
    input_w_float = graph.cast(input_w, dtype=FLOAT32, name="input_w_float", shape=(1,))

    rgb_image = graph.slice_tensor(image, name="rgb_image", shape=("height", "width", 3))
    img_shape = graph.get_shape(rgb_image)  # HW3

    img_h_int = graph.get_item(img_shape, axis=0, idx=0, name="img_h_int", shape=(1,))
    img_h = graph.cast(img_h_int, dtype=FLOAT32, name="img_h", shape=(1,))

    img_w_int = graph.get_item(img_shape, axis=0, idx=1, name="img_w_int", shape=(1,))
    img_w = graph.cast(img_w_int, dtype=FLOAT32, name="img_w", shape=(1,))

    aspect_ratio = graph.div(img_w, img_h, name="aspect_ratio", shape=(1,))

    scalar_1 = _const("one", 1, FLOAT32)
    is_greater = graph.is_greater(aspect_ratio, scalar_1, name="is_greater", shape=(1,))

    then_subgraph = gs.Graph(opset=opset)
    then_img_h = then_subgraph.div(input_w_float, aspect_ratio, name="then_img_h", shape=(1,))
    then_img_w = then_subgraph.identity(input_w_float, name="then_img_w", shape=(1,))
    then_subgraph.outputs = [then_img_h, then_img_w]

    else_subgraph = gs.Graph(opset=opset)
    else_img_h = else_subgraph.identity(input_h_float, name="else_img_h", shape=(1,))
    else_img_w = else_subgraph.mul(input_h_float, aspect_ratio, name="else_img_w", shape=(1,))
    else_subgraph.outputs = [else_img_h, else_img_w]

    new_img_h_float, new_img_w_float = graph.if_statement(
        is_greater,
        then_subgraph,
        else_subgraph,
        dtypes=[FLOAT32, FLOAT32],
        names=["new_img_h_float", "new_img_w_float"],
        shapes=[(1,), (1,)],
    )
    new_img_h = graph.cast(new_img_h_float, dtype=INT64, name="new_img_h", shape=(1,))
    new_img_w = graph.cast(new_img_w_float, dtype=INT64, name="new_img_w", shape=(1,))

    hw_size_float = graph.concat(
        [new_img_h_float, new_img_w_float], axis=0, name="hw_size_float", shape=(2,)
    )
    sizes = graph.cast(hw_size_float, dtype=INT64, name="sizes", shape=(2,))

    resized_img = graph.resize(
        rgb_image, sizes, name="resized_img", shape=("new_img_h", "new_img_w", 3)
    )

    scalar_2 = _const("scalar_2", 2, dtype=INT64)
    pad_x = graph.sub(input_w_int64, new_img_w, name="pad_x", shape=(1,))
    pad_left_float = graph.div(pad_x, scalar_2, name="pad_left_float", shape=(1,))
    pad_left = graph.cast(pad_left_float, dtype=INT64, name="pad_left", shape=(1,))
    pad_right = graph.sub(pad_x, pad_left, name="pad_right", shape=(1,))

    pad_y = graph.sub(input_h_int64, new_img_h, name="pad_y", shape=(1,))
    pad_top_float = graph.div(pad_y, scalar_2, name="pad_top_float", shape=(1,))
    pad_top = graph.cast(pad_top_float, dtype=INT64, name="pad_top", shape=(1,))
    pad_bottom = graph.sub(pad_y, pad_top, name="pad_bottom", shape=(1,))

    padding = [pad_top, pad_left, pad_bottom, pad_right]
    pads_int64 = graph.concat(padding, axis=0, name="padding_int64", shape=(4,))

    padded_img = graph.pad_image(
        resized_img, pads_int64, fill_value, name="padded_img", shape=("input_h", "input_w", 3)
    )

    float_img = graph.cast(
        padded_img, dtype=FLOAT32, name="float_img", shape=("input_h", "input_w", 3)
    )
    scalar_255 = _const("scalar_255", 255, dtype=FLOAT32)
    normalized_img = graph.div(
        float_img, scalar_255, name="normalized_img", shape=("input_h", "input_w", 3)
    )

    transposed_img = graph.transpose(
        normalized_img, perm=[2, 0, 1], name="preprocessed_img", shape=(3, "input_h", "input_w")
    )
    pads = graph.cast(pads_int64, dtype=INT32, name="padding_tlbr", shape=(4,))
    graph.outputs = [transposed_img, pads]
    onnx.save(gs.export_onnx(graph), filepath)


if __name__ == "__main__":
    create_onnx_preprocessing(filepath=str(MODELS_PATH / "preprocessing.onnx"), opset=18)
