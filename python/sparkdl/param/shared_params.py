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

"""
Some parts are copied from pyspark.ml.param.shared and some are complementary
to pyspark.ml.param. The copy is due to some useful pyspark fns/classes being
private APIs.
"""

from functools import wraps
import six

import keras
import tensorflow as tf

from sparkdl.graph.builder import GraphFunction, IsolatedSession
import sparkdl.graph.utils as tfx
from sparkdl.graph.input import TFInputGraph, TFInputGraphBuilder
import sparkdl.utils.keras_model as kmutil


########################################################
# Copied from PySpark for backward compatibility.
# They first appeared in Apache Spark version 2.1.1.
########################################################

def keyword_only(func):
    """
    A decorator that forces keyword arguments in the wrapped method
    and saves actual input keyword arguments in `_input_kwargs`.

    .. note:: Should only be used to wrap a method where first arg is `self`
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError("Method %s forces keyword arguments." % func.__name__)
        self._input_kwargs = kwargs
        return func(self, **kwargs)

    return wrapper


class HasInputCol(Params):
    """
    Mixin for param inputCol: input column name.
    """

    inputCol = Param(
        Params._dummy(), "inputCol", "input column name.", typeConverter=TypeConverters.toString)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def getInputCol(self):
        """
        Gets the value of inputCol or its default value.
        """
        return self.getOrDefault(self.inputCol)


class HasOutputCol(Params):
    """
    Mixin for param outputCol: output column name.
    """

    outputCol = Param(
        Params._dummy(), "outputCol", "output column name.", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasOutputCol, self).__init__()
        self._setDefault(outputCol=self.uid + '__output')

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def getOutputCol(self):
        """
        Gets the value of outputCol or its default value.
        """
        return self.getOrDefault(self.outputCol)

########################################################
# New in sparkdl
########################################################


class HasOutputNodeName(Params):
    # TODO: docs
    outputNodeName = Param(Params._dummy(), "outputNodeName",
                           "name of the graph element/node corresponding to the output",
                           typeConverter=TypeConverters.toString)

    def setOutputNodeName(self, value):
        return self._set(outputNodeName=value)

    def getOutputNodeName(self):
        return self.getOrDefault(self.outputNodeName)


class HasLabelCol(Params):
    """
    When training Keras image models in a supervised learning setting,
    users will provide a :py:obj:`DataFrame` column with the labels.

    .. note:: The Estimator expect this columnd to contain data directly usable for the Keras model.
              This usually means that the labels are already encoded in one-hot format.
              Please consider adding a :py:obj:`OneHotEncoder` to transform the label column.
    """
    labelCol = Param(Params._dummy(), "labelCol",
                     "name of the column storing the training data labels",
                     typeConverter=TypeConverters.toString)

    def setLabelCol(self, value):
        return self._set(labelCol=value)

    def getLabelCol(self):
        return self.getOrDefault(self.labelCol)


class HasKerasModel(Params):
    """
    This parameter allows users to provide Keras model file
    """
    # TODO: add an option to allow user to use Keras Model object
    modelFile = Param(Params._dummy(), "modelFile",
                      "HDF5 file containing the Keras model (architecture and weights)",
                      typeConverter=TypeConverters.toString)

    kerasFitParams = Param(Params._dummy(), "kerasFitParams",
                           "dict with parameters passed to Keras model fit method")

    def __init__(self):
        super(HasKerasModel, self).__init__()
        self._setDefault(kerasFitParams={'verbose': 1})

    def setModelFile(self, value):
        return self._set(modelFile=value)

    def getModelFile(self):
        return self.getOrDefault(self.modelFile)

    def setKerasFitParams(self, value):
        return self._set(kerasFitParams=value)

    def getKerasFitParams(self):
        return self.getOrDefault(self.kerasFitParams)


class HasKerasOptimizer(Params):
    # TODO: docs
    kerasOptimizer = Param(Params._dummy(), "kerasOptimizer",
                           "Name of the optimizer for training a Keras model",
                           typeConverter=SparkDLTypeConverters.toKerasOptimizer)

    def __init__(self):
        super(HasKerasOptimizer, self).__init__()
        # NOTE(phi-dbq): This is the recommended optimizer as of September 2017.
        self._setDefault(kerasOptimizer='adam')

    def setKerasOptimizer(self, value):
        return self._set(kerasOptimizer=value)

    def getKerasOptimizer(self):
        return self.getOrDefault(self.kerasOptimizer)


class HasKerasLoss(Params):
    # TODO: docs
    kerasLoss = Param(Params._dummy(), "kerasLoss",
                      "Name of the loss (objective function) for training a Keras model",
                      typeConverter=SparkDLTypeConverters.toKerasLoss)

    def seKerasLoss(self, value):
        return self._set(kerasLoss=value)

    def getKerasLoss(self):
        return self.getOrDefault(self.kerasLoss)


class HasOutputMapping(Params):
    """
    Mixin for param outputMapping: ordered list of ('outputTensorOpName', 'outputColName') pairs
    """
    outputMapping = Param(
        Params._dummy(),
        "outputMapping",
        "Mapping output :class:`tf.Operation` names to DataFrame column names",
        typeConverter=SparkDLTypeConverters.asTensorToColumnMap)

    def setOutputMapping(self, value):
        # NOTE(phi-dbq): due to the nature of TensorFlow import modes, we can only derive the
        #                serializable TFInputGraph object once the inputMapping and outputMapping
        #                parameters are provided.
        raise NotImplementedError(
            "Please use the Transformer's constructor to assign `outputMapping` field.")

    def getOutputMapping(self):
        return self.getOrDefault(self.outputMapping)


class HasInputMapping(Params):
    """
    Mixin for param inputMapping: ordered list of ('inputColName', 'inputTensorOpName') pairs
    """
    inputMapping = Param(
        Params._dummy(),
        "inputMapping",
        "Mapping input DataFrame column names to :class:`tf.Operation` names",
        typeConverter=SparkDLTypeConverters.asColumnToTensorMap)

    def setInputMapping(self, value):
        # NOTE(phi-dbq): due to the nature of TensorFlow import modes, we can only derive the
        #                serializable TFInputGraph object once the inputMapping and outputMapping
        #                parameters are provided.
        raise NotImplementedError(
            "Please use the Transformer's constructor to assigne `inputMapping` field.")

    def getInputMapping(self):
        return self.getOrDefault(self.inputMapping)


class HasTFInputGraph(Params):
    """
    Mixin for param tfInputGraph: a serializable object derived from a TensorFlow computation graph.
    """
    tfInputGraph = Param(
        Params._dummy(),
        "tfInputGraph",
        "A serializable object derived from a TensorFlow computation graph",
        typeConverter=SparkDLTypeConverters.toTFInputGraph)

    def __init__(self):
        super(HasTFInputGraph, self).__init__()
        self._setDefault(tfInputGraph=None)

    def setTFInputGraph(self, value):
        # NOTE(phi-dbq): due to the nature of TensorFlow import modes, we can only derive the
        #                serializable TFInputGraph object once the inputMapping and outputMapping
        #                parameters are provided.
        raise NotImplementedError(
            "Please use the Transformer's constructor to assign `tfInputGraph` field.")

    def getTFInputGraph(self):
        return self.getOrDefault(self.tfInputGraph)


class HasTFHParams(Params):
    """
    Mixin for TensorFlow model hyper-parameters
    """
    tfHParams = Param(
        Params._dummy(),
        "hparams",
        "instance of :class:`tf.contrib.training.HParams`, a key-value map-like object",
        typeConverter=SparkDLTypeConverters.toTFHParams)

    def setTFHParams(self, value):
        return self._set(tfHParam=value)

    def getTFHParams(self):
        return self.getOrDefault(self.tfHParams)
