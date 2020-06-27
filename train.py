# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    train.py
   Description :
   Author :       Wings DH
   Time：         2020/6/19 1:54 下午
-------------------------------------------------
   Change Activity:
                   2020/6/19: Create
-------------------------------------------------
"""
import os

import tensorflow as tf

from models import bert_modeling, optimization
from models.bert_layers import get_masked_lm_output_without_position

flags = tf.flags
FLAGS = flags.FLAGS


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     loss_lambda=0.5,
                     use_one_hot_embeddings=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        soft_mask_label = features['soft_mask_label']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = bert_modeling.SoftMarkedBertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output_without_position(
            bert_config, model.get_sequence_output(), model.get_embedding_table(), masked_lm_ids, masked_lm_weights)
        masked_lm_loss = tf.identity(masked_lm_loss, name="masked_lm_loss")

        with tf.variable_scope("sigmoid/detect"):
            prob_logits = tf.squeeze(model.prob, axis=-1)
            prob_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=soft_mask_label, logits=prob_logits)
            numerator = tf.reduce_sum(masked_lm_weights * prob_loss)
            denominator = tf.reduce_sum(masked_lm_weights) + 1e-5
            prob_loss = numerator / denominator

        # merge masked_lm_loss of bert and prob_loss from gru
        total_loss = masked_lm_loss * loss_lambda + (1 - loss_lambda) * prob_loss
        total_loss = tf.identity(total_loss, name='total_loss')
        tvars = tf.trainable_variables()

        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
            return output_spec
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(_masked_lm_example_loss, _masked_lm_log_probs, _masked_lm_ids,
                          _masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                _masked_lm_log_probs = tf.reshape(_masked_lm_log_probs,
                                                  [-1, _masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    _masked_lm_log_probs, axis=-1, output_type=tf.int32)
                _masked_lm_example_loss = tf.reshape(_masked_lm_example_loss, [-1])
                _masked_lm_ids = tf.reshape(_masked_lm_ids, [-1])
                _masked_lm_weights = tf.reshape(_masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=_masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=_masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=_masked_lm_example_loss, weights=_masked_lm_weights)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                }

            eval_metrics = metric_fn(
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                masked_lm_weights)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics)

            return output_spec

    return model_fn


def input_fn_builder(input_files,
                     max_seq_length,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        # batch_size = params["batch_size"]
        batch_size = 8

        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_seq_length], tf.float32),
            "soft_mask_label":
                tf.FixedLenFeature([max_seq_length], tf.float32),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        # if t.dtype == tf.int64:
        #     t = tf.to_int32(t)
        example[name] = t

    return example


def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = bert_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        loss_lambda=0.1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=10000,
        num_warmup_steps=1000)

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                        model_dir=FLAGS.output_dir, log_step_count_steps=200)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=100000)


def init_parameters():
    # Required parameters
    flags.DEFINE_string(
        "input_file", None,
        "Input TF example files (can be a glob or comma separated).")
    flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    flags.DEFINE_string("output_dir", None,
                        "The output directory where the model checkpoints will be written.")

    # Optional parameters
    flags.DEFINE_string("task_name", None, "The name of the task to train.")
    flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")
    flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

    flags.DEFINE_integer("save_checkpoints_steps", 1000,
                         "How often to save the model checkpoint.")

    flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded. Must match data generation.")


if __name__ == "__main__":
    init_parameters()
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
