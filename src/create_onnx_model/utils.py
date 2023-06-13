import onnx_graphsurgeon as gs
import numpy as np

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
def add(self, a, b, name="sum", shape=None):
    out = self.layer(op="Add", inputs=[a, b], outputs=[name])[0]
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
