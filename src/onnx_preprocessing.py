import onnx_graphsurgeon as gs
import numpy as np
import onnx

INT32 = np.int32
INT64 = np.int64
UINT8 = np.uint8
FLOAT32 = np.float32
BOOL = np.bool_
STR = np.str_


def _const(name, value, dtype):
    return gs.Constant(name=name, values=np.array([value], dtype=dtype))


def parse_out(out, dtype, name, shape):
    out.dtype = dtype
    out.name = name
    out.shape = shape
    return out


@gs.Graph.register()
def slice_tensor(
    self, tensor, starts: int = 0, ends: int = 3, axes: int = 2, name="slice", shape=None
):
    out = self.layer(
        op="Slice",
        inputs=[
            tensor,
            _const("starts", starts, INT64),
            _const("ends", ends, INT64),
            _const("axes", axes, INT64),
        ],
        outputs=[name],
    )[0]
    return parse_out(out, tensor.dtype, name, shape)


@gs.Graph.register()
def get_shape(self, rgb_image, name="shape", shape=None):
    out = self.layer(op="Shape", inputs=[rgb_image], outputs=[name])[0]
    return parse_out(out, INT64, name, shape)


@gs.Graph.register()
def div(self, a, b, name="ratio", shape=None):
    out = self.layer(op="Div", inputs=[a, b], outputs=[name])[0]
    return parse_out(out, a.dtype, name, shape)


@gs.Graph.register()
def sub(self, a, b, name="diff", shape=None):
    out = self.layer(op="Sub", inputs=[a, b], outputs=[name])[0]
    return parse_out(out, a.dtype, name, shape)


@gs.Graph.register()
def get_item(self, tensor, axis, idx, name="item", shape=None):
    out = self.layer(
        op="Gather",
        attrs={"axis": axis},
        inputs=[tensor, _const(f"index_{idx}", idx, INT64)],
        outputs=[name],
    )[0]
    return parse_out(out, tensor.dtype, name, shape)


@gs.Graph.register()
def cast(self, a, dtype, name="casted", shape=None):
    out = self.layer(op="Cast", attrs={"to": dtype}, inputs=[a], outputs=[name])[0]
    return parse_out(out, dtype, name, shape)


@gs.Graph.register()
def is_greater(self, a, b, name="is_greater", shape=None):
    out = self.layer(op="Greater", inputs=[a, b], outputs=[name])[0]
    return parse_out(out, BOOL, name, shape)


# TODO
@gs.Graph.register()
def if_statement(
    self, cond, then_branch, else_branch, dtypes=[FLOAT32], names=["if_out"], shapes=[None]
):
    outs = self.layer(
        op="If",
        attrs={"then_branch": then_branch, "else_branch": else_branch},
        inputs=[cond],
        outputs=names,
    )
    return [parse_out(outs[i], dtypes[i], names[i], shapes[i]) for i in range(len(outs))]


@gs.Graph.register()
def identity(self, a, name="copy", shape=None):
    out = self.layer(op="Identity", inputs=[a], outputs=[name])[0]
    return parse_out(out, a.dtype, name, shape)


@gs.Graph.register()
def constant(self, a, name="const", shape=None):
    out = self.layer(op="Constant", attrs={"value": a}, outputs=[name])[0]
    return parse_out(out, a.dtype, name, shape)


@gs.Graph.register()
def concat(self, inputs, axis: int = 0, name="concat_results", shape=None):
    out = self.layer(op="Concat", attrs={"axis": axis}, inputs=inputs, outputs=[name])[0]
    return parse_out(out, inputs[0].dtype, name, shape)


@gs.Graph.register()
def resize(self, image, sizes, name="resized", shape=None):
    out = self.layer(
        op="Resize",
        attrs={"mode": "linear", "axes": [0, 1]},
        inputs=[image, _const("", "", STR), _const("", "", STR), sizes],
        outputs=[name],
    )[0]
    return parse_out(out, image.dtype, name, shape)


@gs.Graph.register()
def pad_image(self, image, pads, fill_value, name="padded", shape=None):
    # pads = [pad_top, pad_left, pad_bottom, pad_right]
    out = self.layer(
        op="Pad",
        attrs={"mode": "constant"},
        inputs=[
            image,  # data
            pads,  # pads
            fill_value,  # constant_value
            gs.Constant("pad_axes", np.array([0, 1], dtype=INT64)),  # axes
        ],
        outputs=[name],
    )[0]
    return parse_out(out, image.dtype, name, shape)


@gs.Graph.register()
def transpose(self, tensor, perm=[2, 0, 1], name="transposed", shape=None):
    out = self.layer(op="Transpose", attrs={"perm": perm}, inputs=[tensor], outputs=[name])[0]
    return parse_out(out, tensor.dtype, name, shape)


def create_onnx_preprocessing(filepath: str = "preprocessing.onnx", opset: int = 18):
    graph = gs.Graph(opset=opset)

    input_h = gs.Variable(name="input_h", dtype=INT32, shape=(1,))
    input_w = gs.Variable(name="input_w", dtype=INT32, shape=(1,))
    image = gs.Variable(name="image", dtype=UINT8, shape=("height", "width", "channels"))
    fill_value = gs.Variable(name="fill_value", dtype=UINT8, shape=(1,))

    input_h_int64 = graph.cast(input_h, dtype=INT64, name="input_h_int64", shape=(1,))
    input_w_int64 = graph.cast(input_w, dtype=INT64, name="input_w_int64", shape=(1,))

    input_h_float = graph.cast(input_h, dtype=FLOAT32, name="input_h_float", shape=(1,))
    input_w_float = graph.cast(input_w, dtype=FLOAT32, name="input_w_float", shape=(1,))

    inputs = [image, input_h, input_w, fill_value]
    graph.inputs = inputs

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
    else_img_w = else_subgraph.div(input_h_float, aspect_ratio, name="else_img_w", shape=(1,))
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
        normalized_img, perm=[2, 0, 1], name="preprocessed_img", shape=("input_h", "input_w", 3)
    )
    pads = graph.cast(pads_int64, dtype=INT32, name="padding_tlbr", shape=(4,))
    graph.outputs = [transposed_img, pads]
    onnx.save(gs.export_onnx(graph), filepath)
