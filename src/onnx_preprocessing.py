import onnx_graphsurgeon as gs
import numpy as np
import onnx


def _const(name, value, dtype):
    return gs.Constant(name=name, values=np.array([value], dtype=dtype))


def parse_outs(outs, dtypes, names):
    for i in range(len(outs)):
        outs[i].dtype = dtypes[i]
        outs[i].name = names[i]
    return outs


@gs.Graph.register()
def slice_tensor(
    self, input_image, starts: int = 0, ends: int = 3, axes: int = 2, outputs=["slice"]
):
    outs = self.layer(
        op="Slice",
        inputs=[
            input_image,
            _const("starts", starts, np.int64),
            _const("ends", ends, np.int64),
            _const("axes", axes, np.int64),
        ],
        outputs=outputs,
    )
    return parse_outs(outs, [input_image.dtype], outputs)


@gs.Graph.register()
def shape(self, rgb_image, outputs=["shape"]):
    outs = self.layer(op="Shape", inputs=[rgb_image], outputs=outputs)
    return parse_outs(outs, [np.int64], outputs)


@gs.Graph.register()
def div(self, a, b, outputs=["ratio"]):
    outs = self.layer(op="Div", inputs=[a, b], outputs=outputs)
    return parse_outs(outs, [a.dtype], outputs)


@gs.Graph.register()
def sub(self, a, b, outputs=["diff"]):
    outs = self.layer(op="Sub", inputs=[a, b], outputs=outputs)
    return parse_outs(outs, [a.dtype], outputs)


@gs.Graph.register()
def get_item(self, tensor, axis, idx, outputs=["item"]):
    outs = self.layer(
        op="Gather",
        attrs={"axis": axis},
        inputs=[tensor, _const(f"index_{idx}", idx, np.int64)],
        outputs=outputs,
    )
    return parse_outs(outs, [tensor.dtype], outputs)


@gs.Graph.register()
def cast(self, a, dtype, outputs=["casted"]):
    outs = self.layer(op="Cast", attrs={"to": dtype}, inputs=[a], outputs=outputs)
    return parse_outs(outs, [dtype], outputs)


@gs.Graph.register()
def is_greater(self, a, b, outputs=["is_greater"]):
    outs = self.layer(op="Greater", inputs=[a, b], outputs=outputs)
    return parse_outs(outs, [np.bool_], outputs)


@gs.Graph.register()
def if_statement(self, cond, then_branch, else_branch, dtypes=[np.float32], outputs=["if_out"]):
    outs = self.layer(
        op="If",
        attrs={"then_branch": then_branch, "else_branch": else_branch},
        inputs=[cond],
        outputs=outputs,
    )
    return parse_outs(outs, dtypes, outputs)


@gs.Graph.register()
def identity(self, a, outputs=["copy"]):
    outs = self.layer(op="Identity", inputs=[a], outputs=outputs)
    return parse_outs(outs, [a.dtype], outputs)


@gs.Graph.register()
def constant(self, a, outputs=["const"]):
    outs = self.layer(op="Constant", attrs={"value": a}, outputs=outputs)
    return parse_outs(outs, [a.dtype], outputs)


@gs.Graph.register()
def concat(self, inputs, axis: int = 0, outputs=["concat_results"]):
    outs = self.layer(op="Concat", attrs={"axis": axis}, inputs=inputs, outputs=outputs)
    return parse_outs(outs, [inputs[0].dtype], outputs)


@gs.Graph.register()
def resize(self, image, sizes, outputs=["resized"]):
    outs = self.layer(
        op="Resize",
        attrs={"mode": "linear", "axes": [0, 1]},
        inputs=[image, _const("", "", np.str_), _const("", "", np.str_), sizes],  # roi  # scales
        outputs=outputs,
    )
    return parse_outs(outs, [image.dtype], outputs)


@gs.Graph.register()
def pad_image(self, image, pads, constant_value: int = 114, outputs=["padded"]):
    # pads = [pad_top, pad_left, pad_bottom, pad_right]
    outs = self.layer(
        op="Pad",
        attrs={"mode": "constant"},
        inputs=[
            image,  # data
            pads,  # pads
            _const("constant_value", constant_value, np.uint8),  # constant_value
            gs.Constant("pad_axes", np.array([0, 1], dtype=np.int64)),  # axes
        ],
        outputs=outputs,
    )
    return parse_outs(outs, [image.dtype], outputs)


@gs.Graph.register()
def transpose(self, tensor, perm=[2, 0, 1], outputs=["transposed"]):
    outs = self.layer(op="Transpose", attrs={"perm": perm}, inputs=[tensor], outputs=outputs)
    return parse_outs(outs, [tensor.dtype], outputs)


def create_onnx_preprocessing(filepath: str = "preprocessing.onnx", opset: int = 18):
    graph = gs.Graph(opset=opset)

    input_h_int32 = gs.Variable(name="input_h_int32", dtype=np.int32, shape=(1,))
    input_h = graph.cast(input_h_int32, dtype=np.int64, outputs=["input_h"])
    input_h_float = graph.cast(*input_h, dtype=np.float32, outputs=["input_h_float"])

    input_w_int32 = gs.Variable(name="input_w_int32", dtype=np.int32, shape=(1,))
    input_w = graph.cast(input_w_int32, dtype=np.int64, outputs=["input_w"])
    input_w_float = graph.cast(*input_w, dtype=np.float32, outputs=["input_w_float"])

    input_image = gs.Variable(
        name="input_image", dtype=np.uint8, shape=("height", "width", "channels")
    )
    inputs = [input_image, input_h_int32, input_w_int32]
    graph.inputs = inputs

    rgb_image = graph.slice_tensor(input_image, outputs=["X"])
    img_shape = graph.shape(*rgb_image)  # HWC

    img_h_int = graph.get_item(*img_shape, axis=0, idx=0, outputs=["img_h_int"])
    img_h = graph.cast(*img_h_int, dtype=np.float32, outputs=["img_h"])

    img_w_int = graph.get_item(*img_shape, axis=0, idx=1, outputs=["img_w_int"])
    img_w = graph.cast(*img_w_int, dtype=np.float32, outputs=["img_w"])

    aspect_ratio = graph.div(*img_w, *img_h, outputs=["aspect_ratio"])

    is_greater = graph.is_greater(
        *aspect_ratio, _const("one", 1, np.float32), outputs=["is_greater"]
    )

    then_subgraph = gs.Graph(opset=opset)
    then_img_h = then_subgraph.div(*input_w_float, *aspect_ratio, outputs=["then_img_h"])
    then_img_w = then_subgraph.identity(*input_w_float, outputs=["then_img_w"])
    then_subgraph.outputs = then_img_h + then_img_w

    else_subgraph = gs.Graph(opset=opset)
    else_img_h = else_subgraph.identity(*input_h_float, outputs=["else_img_h"])
    else_img_w = else_subgraph.div(*input_h_float, *aspect_ratio, outputs=["else_img_w"])
    else_subgraph.outputs = else_img_h + else_img_w

    new_img_h_float, new_img_w_float = graph.if_statement(
        *is_greater,
        then_subgraph,
        else_subgraph,
        dtypes=[np.float32, np.float32],
        outputs=["new_img_h_float", "new_img_w_float"],
    )
    new_img_h = graph.cast(new_img_h_float, dtype=np.int64, outputs=["new_img_h"])
    new_img_w = graph.cast(new_img_w_float, dtype=np.int64, outputs=["new_img_w"])

    hw_size_float = graph.concat(
        [new_img_h_float, new_img_w_float], axis=0, outputs=["hw_size_float"]
    )
    sizes = graph.cast(*hw_size_float, dtype=np.int64, outputs=["sizes"])

    resized_img = graph.resize(*rgb_image, *sizes, outputs=["resized_img"])

    scalar_2 = _const("scalar_2", 2, dtype=np.int64)
    pad_x = graph.sub(*input_w, *new_img_w, outputs=["pad_x"])
    pad_left_float = graph.div(*pad_x, scalar_2, outputs=["pad_left_float"])
    pad_left = graph.cast(*pad_left_float, np.int64, outputs=["pad_left"])
    pad_right = graph.sub(*pad_x, *pad_left, outputs=["pad_right"])

    pad_y = graph.sub(*input_h, *new_img_h, outputs=["pad_y"])
    pad_top_float = graph.div(*pad_y, scalar_2, outputs=["pad_top_float"])
    pad_top = graph.cast(*pad_top_float, np.int64, outputs=["pad_top"])
    pad_bottom = graph.sub(*pad_y, *pad_top, outputs=["pad_bottom"])

    padding = pad_top + pad_left + pad_bottom + pad_right
    pads = graph.concat(padding, axis=0, outputs=["pads"])
    padded_img = graph.pad_image(*resized_img, *pads, constant_value=114, outputs=["padded_img"])

    float_img = graph.cast(*padded_img, np.float32, outputs=["float_img"])
    scalar_255 = _const("scalar_255", 255, dtype=np.float32)
    normalized_img = graph.div(*float_img, scalar_255, outputs=["normalized_img"])

    transposed_img = graph.transpose(*normalized_img, perm=[2, 0, 1], outputs=["preprocessed_img"])

    graph.outputs = transposed_img
    onnx.save(gs.export_onnx(graph), filepath)
