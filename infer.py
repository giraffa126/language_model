# coding: utf8
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
from lm_model import BiRNNLM 
from reader import DataReader

def infer(args):
    model = BiRNNLM(vocab_size=args.vocab_size)
    saver = tf.train.Saver()
    init = tf.group(tf.global_variables_initializer(), 
            tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint(args.model_path))
        test_reader = DataReader(args.vocab_path, args.test_data_path, 
                args.vocab_size, args.batch_size)
        for test_batch in test_reader.batch_generator():
            inputs, outputs = test_batch
            _loss = sess.run(model.loss, 
                    feed_dict={model.inputs: inputs, model.outputs: outputs})
            ppl = np.exp(_loss)
            print(ppl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="./output")
    parser.add_argument("--test_data_path", type=str, default="./data/ptb.valid.txt")
    parser.add_argument("--vocab_path", type=str, default="./data/vocab.pkl")
    args = parser.parse_args()
    infer(args)
