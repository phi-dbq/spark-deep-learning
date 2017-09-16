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

import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2

import sparkdl.graph.utils as tfx

__all__ = ["TFInputGraph"]


class TFInputGraph(object):
    """
    An opaque serializable object containing TensorFlow graph.

    [WARNING] This class should not be called by any user code.
    """
    def __init__(self):
        raise NotImplementedError(
            "Please do NOT construct TFInputGraph directly. Instead, use one of the helper functions")

    @classmethod
    def _new_obj_internal(cls):
        # pylint: disable=attribute-defined-outside-init
        obj = object.__new__(cls)
        # TODO: for (de-)serialization, the class should correspond to a ProtocolBuffer definition.
        obj.graph_def = None
        obj.input_tensor_name_from_signature = None
        obj.output_tensor_name_from_signature = None
        return obj

    @classmethod
    def fromGraph(cls, graph, sess, feed_names, fetch_names):
        """
        Construct a TFInputGraphBuilder from a in memory tf.Graph object
        """
        assert isinstance(graph, tf.Graph), \
            ('expect tf.Graph type but got', type(graph))

        def import_graph_fn(_sess):
            assert _sess == sess, 'must have the same session'
            return _GinBuilderInfo()

        return _GinBuilder(import_graph_fn, sess, graph).build(feed_names, fetch_names)

    @classmethod
    def fromGraphDef(cls, graph_def, feed_names, fetch_names):
        """
        Construct a TFInputGraphBuilder from a tf.GraphDef object
        """
        assert isinstance(graph_def, tf.GraphDef), \
            ('expect tf.GraphDef type but got', type(graph_def))

        def import_graph_fn(sess):
            with sess.as_default():
                tf.import_graph_def(graph_def, name='')
            return _GinBuilderInfo()

        return _GinBuilder(import_graph_fn).build(feed_names, fetch_names)

    @classmethod
    def fromCheckpoint(cls, checkpoint_dir):
        return cls._from_checkpoint_impl(checkpoint_dir, signature_def_key=None)

    @classmethod
    def fromCheckpointWithSignature(cls, checkpoint_dir, signature_def_key):
        assert signature_def_key is not None
        return cls._from_checkpoint_impl(checkpoint_dir, signature_def_key)

    @classmethod
    def fromSavedModel(cls, saved_model_dir, tag_set):
        return cls._from_saved_model_impl(saved_model_dir, tag_set, signature_def_key=None)

    @classmethod
    def fromSavedModelWithSignature(cls, saved_model_dir, tag_set, signature_def_key):
        assert signature_def_key is not None
        return cls._from_saved_model_impl(saved_model_dir, tag_set, signature_def_key)

    @classmethod
    def _from_checkpoint_impl(cls, checkpoint_dir, signature_def_key=None):
        """
        Construct a TFInputGraphBuilder from a model checkpoint
        """
        def import_graph_fn(sess):
            # Load checkpoint and import the graph
            with sess.as_default():
                ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)

                # NOTE(phi-dbq): we must manually load meta_graph_def to get the signature_def
                #                the current `import_graph_def` function seems to ignore
                #                any signature_def fields in a checkpoint's meta_graph_def.
                meta_graph_def = meta_graph_pb2.MetaGraphDef()
                with open("{}.meta".format(ckpt_path), 'rb') as fin:
                    meta_graph_def.ParseFromString(fin.read())

                saver = tf.train.import_meta_graph(meta_graph_def, clear_devices=True)
                saver.restore(sess, ckpt_path)

                sig_def = None
                if signature_def_key is not None:
                    sig_def = meta_graph_def.signature_def[signature_def_key]
                    assert sig_def, 'singnature_def_key {} provided, '.format(signature_def_key) + \
                        'but failed to find it from the meta_graph_def ' + \
                        'from checkpoint {}'.format(checkpoint_dir)

            return _GinBuilderInfo(sig_def=sig_def)

        return _GinBuilder(import_graph_fn).build()

    @classmethod
    def _from_saved_model_impl(cls, saved_model_dir, tag_set, signature_def_key=None):
        """
        Construct a TFInputGraphBuilder from a SavedModel
        """
        def import_graph_fn(sess):
            tag_sets = tag_set.split(',')
            meta_graph_def = tf.saved_model.loader.load(sess, tag_sets, saved_model_dir)

            sig_def = None
            if signature_def_key is not None:
                sig_def = tf.contrib.saved_model.get_signature_def_by_key(
                    meta_graph_def, signature_def_key)

            return _GinBuilderInfo(sig_def=sig_def)

        return _GinBuilder(import_graph_fn).build()


class _GinBuilderInfo(object):
    def __init__(self, sig_def=None):
        self.sig_def = sig_def


class _GinBuilder(object):
    def __init__(self, import_graph_fn, sess=None, graph=None):
        self.import_graph_fn = import_graph_fn
        assert (sess is None) == (graph is None)
        self.graph = graph or tf.Graph()
        if sess is not None:
            self.sess = sess
            self._should_clean = True
        else:
            self.sess = tf.Session(self.graph)
            self._should_clean = False

    def _build_impl(self, feed_names, fetch_names):
        # pylint: disable=protected-access,attribute-defined-outside-init
        gin = TFInputGraph._new_obj_internal()
        assert (feed_names is None) == (fetch_names is None)
        must_have_sig_def = fetch_names is None
        with self.sess.as_default():
            _ginfo = self.import_graph_fn(self.sess)
            # TODO: extract signature mappings
            if must_have_sig_def:
                raise NotImplementedError("cannot extract mappings from sig_def at the moment")
            for tnsr_name in feed_names:
                assert tfx.get_op(self.graph, tnsr_name)
            fetches = [tfx.get_tensor(self.graph, tnsr_name) for tnsr_name in fetch_names]
            gin.graph_def = tfx.strip_and_freeze_until(fetches, self.graph, self.sess)
        return gin

    def build(self, feed_names=None, fetch_names=None):
        try:
            gin = self._build_impl(feed_names, fetch_names)
        finally:
            if self._should_clean:
                self.sess.close()
        return gin

# def the_rest(input_mapping, output_mapping):
#     graph = tf.Graph()
#     with tf.Session(graph=graph) as sess:
#         # Append feeds and input mapping
#         _input_mapping = {}
#         if isinstance(input_mapping, dict):
#             input_mapping = input_mapping.items()
#         for input_colname, tnsr_or_sig in input_mapping:
#             if sig_def:
#                 tnsr = sig_def.inputs[tnsr_or_sig].name
#             else:
#                 tnsr = tnsr_or_sig
#             _input_mapping[input_colname] = tfx.op_name(graph, tnsr)
#         input_mapping = _input_mapping

#         # Append fetches and output mapping
#         fetches = []
#         _output_mapping = {}
#         # By default the output columns will have the name of their
#         # corresponding `tf.Graph` operation names.
#         # We have to convert them to the user specified output names
#         if isinstance(output_mapping, dict):
#             output_mapping = output_mapping.items()
#         for tnsr_or_sig, requested_colname in output_mapping:
#             if sig_def:
#                 tnsr = sig_def.outputs[tnsr_or_sig].name
#             else:
#                 tnsr = tnsr_or_sig
#             fetches.append(tfx.get_tensor(graph, tnsr))
#             tf_output_colname = tfx.op_name(graph, tnsr)
#             # NOTE(phi-dbq): put the check here as it will be the entry point to construct
#             #                a `TFInputGraph` object.
#             assert tf_output_colname not in _output_mapping, \
#                 "operation {} has multiple output tensors and ".format(tf_output_colname) + \
#                 "at least two of them are used in the output DataFrame. " + \
#                 "Operation names are used to name columns which leads to conflicts. "  + \
#                 "You can apply `tf.identity` ops to each to avoid name conflicts."
#             _output_mapping[tf_output_colname] = requested_colname
#         output_mapping = _output_mapping

#         gdef = tfx.strip_and_freeze_until(fetches, graph, sess)

#     return TFInputGraph(gdef), input_mapping, output_mapping
