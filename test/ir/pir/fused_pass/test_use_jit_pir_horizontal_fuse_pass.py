# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


import unittest

import numpy as np

import paddle
from paddle import nn


class Matmul(nn.Layer):
    def __init__(self, in_features, out_features, weight_attr=None):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[in_features, out_features], attr=weight_attr, dtype="float32"
        )

    def forward(self, x):
        return paddle.matmul(x, self.weight)


class MatmulHorizontalLayer(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, num_layers=32, weight_attr=None):
        super().__init__()
        self.layers = nn.LayerList(
            [
                Matmul(hidden_size, intermediate_size, weight_attr)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        results = []
        for layer in self.layers:
            result = layer(x)
            results.append(result)
        return results


class TestMatmulHorizontalFusePattern(unittest.TestCase):
    def setUp(self):
        self.bsz = 2
        self.seq_len = 16
        self.num_head = 2
        self.head_dim = 16
        self.hidden_size = self.num_head * self.head_dim
        self.intermediate_size = self.hidden_size
        self.weight_attr = None

    def test_matmul_horizontal_fuse(self):
        x = paddle.randn(shape=[self.bsz, self.seq_len, self.hidden_size])
        layer = MatmulHorizontalLayer(self.hidden_size, self.intermediate_size)
        baseline_results = layer(x)

        optimized_layer = paddle.incubate.jit.inference(
            layer,
            enable_new_ir=True,
            save_model_dir="./tmp/dit",
            exp_enable_use_cutlass=True,
        )
        optimized_results = optimized_layer(x)
        self.verify_results(baseline_results, optimized_results)

    @staticmethod
    def verify_results(expected, actual, atol=1e-5, rtol=1e-5):
        assert len(expected) == len(
            actual
        ), f"Length mismatch: expected {len(expected)}, got {len(actual)}"
        for exp, act in zip(expected, actual):
            assert (
                exp.shape == act.shape
            ), f"Shape mismatch: expected {exp.shape}, got {act.shape}"
            np.testing.assert_allclose(exp.numpy(), act.numpy(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
