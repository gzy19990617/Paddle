# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    add_elementwise_layer,
    broadcast,
    convert_trt_weights_to_tensor,
    get_axes_for_reduce_op,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.add", trt_version="8.x")
@converter_registry.register("pd_op.add_", trt_version="8.x")
def add_converter(network, paddle_op, inputs):
    return add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.SUM
    )


@converter_registry.register("pd_op.scale", trt_version="8.x")
def scale_converter(network, paddle_op, inputs):
    scale = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    bias = paddle_op.attrs().get("bias", 0.0)
    power = paddle_op.attrs().get("power", 1.0)

    # Convert scale, bias, and power to TensorRT weights
    scale_weight = trt.Weights(np.array([scale], dtype=np.float32))
    bias_weight = trt.Weights(np.array([bias], dtype=np.float32))
    power_weight = trt.Weights(np.array([power], dtype=np.float32))

    scale_layer = network.add_scale(
        inputs[0],
        mode=trt.ScaleMode.UNIFORM,
        shift=bias_weight,
        scale=scale_weight,
        power=power_weight,
    )
    return scale_layer.get_output(0)


@converter_registry.register("pd_op.max", trt_version="8.x")
def max_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    axis = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    input_shape = paddle_op.operands()[0].source().shape
    keepdim = paddle_op.attrs()["keepdim"]
    if network.has_implicit_batch_dimension:
        assert (
            axis != 0
        ), "can't reduce on axis == 0 when network has implicit batch dimension"
    output_shape = []
    if len(axis) == 0:
        axis = list(range(len(input_shape)))
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] = len(input_shape) + axis[i]
    layer = network.add_reduce(
        input_tensor,
        trt.ReduceOperation.MAX,
        axes=get_axes_for_reduce_op(axis),
        keep_dims=keepdim,
    )
    return layer.get_output(0)


@converter_registry.register("pd_op.divide", trt_version="8.x")
def divide_converter(network, paddle_op, inputs):
    return add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.DIV
    )


@converter_registry.register("pd_op.subtract", trt_version="8.x")
def substract_converter(network, paddle_op, inputs):
    return add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.SUB
    )


@converter_registry.register("pd_op.multiply", trt_version="8.x")
def multiply_converter(network, paddle_op, inputs):
    return add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.PROD
    )


@converter_registry.register("pd_op.remainder", trt_version="8.x")
@converter_registry.register("pd_op.remainder_", trt_version="8.x")
def remainder_converter(network, paddle_op, inputs):
    weight_shape = paddle_op.operands()[1].source().shape
    input_shape = paddle_op.operands()[0].source().shape

    weight_tensor = inputs[1]
    input_tensor = inputs[0]

    input_tensor = convert_trt_weights_to_tensor(
        network, inputs[0], input_shape
    )
    weight_tensor = convert_trt_weights_to_tensor(
        network, inputs[1], weight_shape
    )

    lhs_val, rhs_val = broadcast(
        network,
        input_tensor,
        weight_tensor,
        input_tensor.name,
        weight_tensor.name,
    )
    # Floor division
    quotient = network.add_elementwise(
        lhs_val, rhs_val, trt.ElementWiseOperation.FLOOR_DIV
    ).get_output(0)

    # Multiply rhs by the quotient
    product = network.add_elementwise(
        rhs_val, quotient, trt.ElementWiseOperation.PROD
    ).get_output(0)

    # Subtract the product from lhs to get the remainder
    remainder = network.add_elementwise(
        lhs_val, product, trt.ElementWiseOperation.SUB
    ).get_output(0)

    return remainder
