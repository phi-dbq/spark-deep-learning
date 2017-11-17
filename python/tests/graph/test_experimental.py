# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

import sparkdl.graph.utils as tfx
from .base_experiment_generators import (
    TestGraphInputBasics, TestGraphInputWithSavedModel, TestGraphInputWithCheckpoint)

##===============================================

_serving_tag = "serving_tag"
_serving_sigdef_key = 'prediction_signature'
# The name of the input tensor
_tensor_input_name = "input_tensor"
# The name of the output tensor (scalar)
_tensor_output_name = "output_tensor"
# The name of the variable
_tensor_var_name = "someVariableName"
# The size of the input tensor
_tensor_size = 3


class IntGraphTest(TestGraphInputBasics, TestGraphInputWithSavedModel, TestGraphInputWithCheckpoint):
    __test__ = True

    def build_graph(self, session):
        x = tf.placeholder(tf.int32, shape=[_tensor_size], name=_tensor_input_name)
        w = tf.Variable(tf.ones(shape=[_tensor_size], dtype=tf.int32), name=_tensor_var_name)
        _ = tf.reduce_max(x * w, name=_tensor_output_name)
        session.run(w.initializer)

    def build_random_data_and_results(self):
        return np.array([1, 2, 3]), 3


class FloatGraphTest(TestGraphInputBasics, TestGraphInputWithSavedModel, TestGraphInputWithCheckpoint):
    __test__ = True

    def build_graph(self, session):
        x = tf.placeholder(tf.float32, shape=[_tensor_size], name=_tensor_input_name)
        w = tf.Variable(tf.ones(shape=[_tensor_size], dtype=tf.float32), name=_tensor_var_name)
        _ = tf.reduce_mean(x * w, name=_tensor_output_name)
        session.run(w.initializer)

    def build_random_data_and_results(self):
        return np.array([1.0, 2.0, 3.0], dtype=np.float32), 2.0


class FloatNoVarGraphTest(TestGraphInputBasics, TestGraphInputWithSavedModel):
    __test__ = True

    def build_graph(self, session):
        x = tf.placeholder(tf.float32, shape=[_tensor_size], name=_tensor_input_name)
        _ = tf.reduce_mean(x + x, name=_tensor_output_name)

    def build_random_data_and_results(self):
        return np.array([1.0, 2.0, 3.0], dtype=np.float32), 4.0
