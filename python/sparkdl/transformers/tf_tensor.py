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

import logging
import tensorflow as tf
import tensorframes as tfs

from pyspark.ml import Transformer

import sparkdl.graph.utils as tfx
from sparkdl.graph.input import TFInputGraph, TFInputGraphBuilder
from sparkdl.param import (keyword_only, SparkDLTypeConverters, HasInputMapping,
                           HasOutputMapping, HasTFInputGraph, HasTFHParams)

__all__ = ['TFTransformer']

logger = logging.getLogger('sparkdl')

class TFTransformer(Transformer, HasTFInputGraph, HasTFHParams, HasInputMapping, HasOutputMapping):
    """
    Applies the TensorFlow graph to the array column in DataFrame.

    Restrictions of the current API:

    We assume that
    - All graphs have a "minibatch" dimension (i.e. an unknown leading
      dimension) in the tensor shapes.
    - Input DataFrame has an array column where all elements have the same length
    """

    @keyword_only
    def __init__(self, tfInputGraph=None, inputMapping=None, outputMapping=None, tfHParms=None):
        """
        __init__(self, tfInputGraph=None, inputMapping=None, outputMapping=None, tfHParms=None)
        """
        super(TFTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, tfInputGraph=None, inputMapping=None, outputMapping=None, tfHParms=None):
        """
        setParams(self, tfInputGraph=None, inputMapping=None, outputMapping=None, tfHParms=None)
        """
        super(TFTransformer, self).__init__()
        kwargs = self._input_kwargs
        # The set of parameters either come from some helper functions,
        # in which case type(_maybe_gin) is already TFInputGraph.
        _maybe_gin = tfInputGraph
        if isinstance(_maybe_gin, TFInputGraph):
            return self._set(**kwargs)

        # Otherwise, `_maybe_gin` needs to be converted to TFInputGraph
        # We put all the conversion logic here rather than in SparkDLTypeConverters
        if isinstance(_maybe_gin, TFInputGraphBuilder):
            gin = _maybe_gin
        elif isinstance(_maybe_gin, tf.Graph):
            gin = TFInputGraphBuilder.fromGraph(_maybe_gin)
        elif isinstance(_maybe_gin, tf.GraphDef):
            gin = TFInputGraphBuilder.fromGraphDef(_maybe_gin)
        else:
            raise TypeError("TFTransformer expect tfInputGraph convertible to TFInputGraph, " + \
                            "but the given type {} cannot be converted, ".format(type(tfInputGraph)) + \
                            "please provide `tf.Graph`, `tf.GraphDef` or use one of the " + \
                            "`get_params_from_*` helper functions to build parameters")

        gin, input_mapping, output_mapping = gin.build(inputMapping, outputMapping)
        kwargs['tfInputGraph'] = gin
        kwargs['inputMapping'] = input_mapping
        kwargs['outputMapping'] = output_mapping

        # Further conanonicalization, e.g. converting dict to sorted str pairs happens here
        return self._set(**kwargs)

    def _transform(self, dataset):
        gin = self.getTFInputGraph()
        input_mapping = self.getInputMapping()
        output_mapping = self.getOutputMapping()

        graph = tf.Graph()
        with tf.Session(graph=graph):
            analyzed_df = tfs.analyze(dataset)

            out_tnsr_op_names = [tfx.as_op_name(tnsr_op_name) for tnsr_op_name, _ in output_mapping]
            tf.import_graph_def(graph_def=gin.graph_def, name='', return_elements=out_tnsr_op_names)

            feed_dict = dict((tfx.op_name(graph, tnsr_op_name), col_name)
                             for col_name, tnsr_op_name in input_mapping)
            fetches = [tfx.get_tensor(graph, tnsr_op_name) for tnsr_op_name in out_tnsr_op_names]

            out_df = tfs.map_blocks(fetches, analyzed_df, feed_dict=feed_dict)

            # We still have to rename output columns
            for old_colname, new_colname in output_mapping:
                if old_colname != new_colname:
                    out_df = out_df.withColumnRenamed(old_colname, new_colname)

        return out_df
