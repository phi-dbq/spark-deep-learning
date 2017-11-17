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

import abc
import contextlib
import glob
import os
import shutil
import tempfile

import numpy as np
import six
import tensorflow as tf

import sparkdl.graph.utils as tfx
from sparkdl.graph.input import TFInputGraph


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


@six.add_metaclass(abc.ABCMeta)
class TestGraphImport(object):
    __test__ = False

    @abc.abstractmethod
    def build_graph(self, session):
        raise NotImplementedError()

    @abc.abstractmethod
    def build_random_data_and_results(self):
        raise NotImplementedError()

    def _check_results(self, gin):
        dataset, expected = self.build_random_data_and_results()
        graph = tf.Graph()
        graph_def = gin.graph_def
        with tf.Session(graph=graph) as sess:
            tf.import_graph_def(graph_def, name="")
            tgt_feed = tfx.get_tensor(_tensor_input_name, graph)
            tgt_fetch = tfx.get_tensor(_tensor_output_name, graph)
            # Run on the testing target
            tgt_out = sess.run(tgt_fetch, feed_dict={tgt_feed: dataset})
            # Working on integers, the calculation should be exact
            assert np.all(tgt_out == expected), (tgt_out, expected)

    def _build_graph_input(self, gin_function):
        """
        Makes a session and a default graph, loads the simple graph into it that contains a variable,
        and then calls gin_function(session) to return the graph input object
        """
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess, graph.as_default():
            self.build_graph(sess)
            return gin_function(sess)

##===============================================

class TestGraphInputBasics(TestGraphImport):

    def test_graph(self):

        def gin_fun(session):
            return TFInputGraph.fromGraph(session.graph, session,
                                          [_tensor_input_name],
                                          [_tensor_output_name])

        gin = self._build_graph_input(gin_fun)
        self._check_results(gin)

    def test_graphdef(self):

        def gin_fun(session):
            with session.as_default():
                fetches = [tfx.get_tensor(_tensor_output_name, session.graph)]
                graph_def = tfx.strip_and_freeze_until(fetches, session.graph, session)

            return TFInputGraph.fromGraphDef(graph_def,
                                             [_tensor_input_name],
                                             [_tensor_output_name])

        gin = self._build_graph_input(gin_fun)
        self._check_results(gin)


class TestGraphInputWithSavedModel(TestGraphImport):

    def _convert_into_saved_model(self, session, saved_model_dir):
        """
        Saves a model in a file. The graph is assumed to be generated with _build_graph_novar.
        """
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        input_tensor = tfx.get_tensor(_tensor_input_name, session.graph)
        output_tensor = tfx.get_tensor(_tensor_output_name, session.graph)
        sig_inputs = {'input_sig': tf.saved_model.utils.build_tensor_info(input_tensor)}
        sig_outputs = {'output_sig': tf.saved_model.utils.build_tensor_info(output_tensor)}
        serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=sig_inputs, outputs=sig_outputs)

        builder.add_meta_graph_and_variables(
            session, [_serving_tag], signature_def_map={_serving_sigdef_key: serving_sigdef})
        builder.save()

    def test_saved_model_with_sigdef(self):
        with _make_temp_directory() as tmp_dir:
            saved_model_dir = os.path.join(tmp_dir, 'saved_model')

            def gin_fun(session):
                # Add saved model parameters
                self._convert_into_saved_model(session, saved_model_dir)
                # Build the transformer from exported serving model
                # We are using signatures, thus must provide the keys
                return TFInputGraph.fromSavedModelWithSignature(saved_model_dir,
                                                                _serving_tag,
                                                                _serving_sigdef_key)

            gin = self._build_graph_input(gin_fun)
            self._check_results(gin)

    def test_saved_model_no_sigdef(self):
        with _make_temp_directory() as tmp_dir:
            saved_model_dir = os.path.join(tmp_dir, 'saved_model')

            def gin_fun(session):
                # Add saved model parameters
                self._convert_into_saved_model(session, saved_model_dir)
                # Build the transformer from exported serving model
                # We are using signatures, thus must provide the keys
                return TFInputGraph.fromSavedModel(saved_model_dir,
                                                   _serving_tag,
                                                   [_tensor_input_name],
                                                   [_tensor_output_name])

            gin = self._build_graph_input(gin_fun)
            self._check_results(gin)


class TestGraphInputWithCheckpoint(TestGraphImport):

    def _convert_into_checkpointed_model(self, session, tmp_dir):
        """
        Writes a model checkpoint in the given directory. The graph is assumed to be generated
         with _build_graph_var.
        """
        ckpt_path_prefix = os.path.join(tmp_dir, 'model_ckpt')
        input_tensor = tfx.get_tensor(_tensor_input_name, session.graph)
        output_tensor = tfx.get_tensor(_tensor_output_name, session.graph)
        var_list = [tfx.get_tensor(_tensor_var_name, session.graph)]
        saver = tf.train.Saver(var_list=var_list)
        _ = saver.save(session, ckpt_path_prefix, global_step=2702)
        sig_inputs = {'input_sig': tf.saved_model.utils.build_tensor_info(input_tensor)}
        sig_outputs = {'output_sig': tf.saved_model.utils.build_tensor_info(output_tensor)}
        serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=sig_inputs, outputs=sig_outputs)

        # A rather contrived way to add signature def to a meta_graph
        meta_graph_def = tf.train.export_meta_graph()

        # Find the meta_graph file (there should be only one)
        _ckpt_meta_fpaths = glob.glob('{}/*.meta'.format(tmp_dir))
        assert len(_ckpt_meta_fpaths) == 1, \
            'expected only one meta graph, but got {}'.format(','.join(_ckpt_meta_fpaths))
        ckpt_meta_fpath = _ckpt_meta_fpaths[0]

        # Add signature_def to the meta_graph and serialize it
        # This will overwrite the existing meta_graph_def file
        meta_graph_def.signature_def[_serving_sigdef_key].CopyFrom(serving_sigdef)
        with open(ckpt_meta_fpath, mode='wb') as fout:
            fout.write(meta_graph_def.SerializeToString())

    def test_checkpoint_with_sigdef(self):
        with _make_temp_directory() as tmp_dir:

            def gin_fun(session):
                self._convert_into_checkpointed_model(session, tmp_dir)
                return TFInputGraph.fromCheckpointWithSignature(tmp_dir,
                                                                _serving_sigdef_key)

            gin = self._build_graph_input(gin_fun)
            self._check_results(gin)

    def test_checkpoint_without_sigdef(self):
        with _make_temp_directory() as tmp_dir:

            def gin_fun(session):
                self._convert_into_checkpointed_model(session, tmp_dir)
                return TFInputGraph.fromCheckpoint(tmp_dir,
                                                   [_tensor_input_name],
                                                   [_tensor_output_name])

            gin = self._build_graph_input(gin_fun)
            self._check_results(gin)

##===============================================

@contextlib.contextmanager
def _make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
