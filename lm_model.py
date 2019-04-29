# coding: utf8
import sys
import os
import numpy as np
import tensorflow as tf

class BiRNNLM(object):
    def __init__(self, vocab_size=200000, emb_size=128, 
            hidden_size=128, num_layers=1, keep_prob=0.5):
        # inputs
        self._vocab_size = vocab_size
        self._emb_size = emb_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._keop_prob = keep_prob
        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.outputs = tf.placeholder(tf.int32, [None, None], name="outputs")
        self.seq_len = tf.reduce_sum(tf.sign(self.inputs), 1)
        # embedding
        self._build_embedding()
        # model
        self._build_model("Encode")
        # loss
        self._build_loss()

    def _build_embedding(self):
        with tf.variable_scope("embedding", tf.AUTO_REUSE):
            self._embedding = tf.get_variable(name="embedding", 
                    initializer=tf.random_uniform([self._vocab_size, self._emb_size], -1.0, 1.0))
            self._input_embs = tf.nn.embedding_lookup(self._embedding, self.inputs)

    def _build_model(self, name=None):
        with tf.variable_scope("birnn_%s" % name, reuse=tf.AUTO_REUSE):
            def make_cell():
                cell = tf.nn.rnn_cell.GRUCell(self._hidden_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._keop_prob)
                return cell
            with tf.variable_scope("fw", reuse=tf.AUTO_REUSE):
                fw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(self._num_layers)])
            with tf.variable_scope("bw", reuse=tf.AUTO_REUSE):
                bw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(self._num_layers)])
            with tf.variable_scope("fw-bw", reuse=tf.AUTO_REUSE):
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                        self._input_embs, sequence_length=self.seq_len, dtype=tf.float32)
            fw_outputs = rnn_outputs[0][:, :-2, :]
            bw_outputs = rnn_outputs[1][:, 2:, :]
            merged_output = tf.concat([fw_outputs, bw_outputs], axis=2)
            #merged_output = tf.transpose(merged_output, [1, 0, 2])
            #rnn_out = merged_output[-1]
            self.logits = tf.layers.dense(merged_output, self._vocab_size)

    def _build_loss(self):
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, 
                    targets=self.outputs,
                    weights=tf.sequence_mask(self.seq_len - 2, tf.shape(self.inputs)[1] - 2, dtype=tf.float32)
            )

