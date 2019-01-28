import tensorflow as tf
import numpy as np
import argparse
import sys
import os
import datetime
import codecs
import time

from tensorflow.python import debug as tf_debug
from graph import *

if __name__ != '__main__':
    sys.exit()

"""
arguments of this script
model_dir: 
n_cpu_cores: 
"""
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', required=True,
    help="Path of the directory which has model_config.py (required)")
parser.add_argument('--n_cpu_cores', default=4, type=int,
    help="Number of CPU cores this script can use when preprocessing dataset")
args = parser.parse_args()

sys.path.insert(0, args.model_dir)
import model_config
print("model_config has been loaded from {}".format(model_config.__file__))

from model_config import Hyperparams as hp
from model_config import Config as conf
import inference

# load dataset
def _file_to_ID_seq(filename, vocab_file_name):
    """load file into dataset"""
    table = tf.contrib.lookup.index_table_from_file(
                vocab_file_name,
                num_oov_buckets=0,
                default_value=conf.UNK_ID,
                key_column_index=0)
    ncpu = args.n_cpu_cores
    return tf.data.TextLineDataset(filename)\
    .map(lambda line: tf.string_split([line]).values, ncpu)\
    .map(lambda tokens: tf.cast(table.lookup(tokens), tf.int32), ncpu)\
    .map(lambda seq: tf.concat( #adding EOS ID
        [seq, tf.ones([1], tf.int32)*model_config.Config.EOS_ID], axis=0), ncpu)\
    .map(lambda seq: (seq, tf.shape(seq)[0]), ncpu)
#    return tf.data.TextLineDataset(filename)\
#    .map(lambda line: tf.string_split([line]).values, ncpu)\
#    .map(lambda tokens: tf.cast(table.lookup(tokens), tf.int32), ncpu)\
#    .map(lambda seq: tf.concat( #adding EOS ID
#        [seq, tf.ones([1], tf.int32)*model_config.Config.EOS_ID], axis=0), ncpu)

#train dataset
train_source = _file_to_ID_seq(model_config.Config.source_train_tok,
                               model_config.Config.vocab_source)
train_target = _file_to_ID_seq(model_config.Config.target_train_tok,
                               model_config.Config.vocab_target)
train_data = tf.data.Dataset.zip((train_source, train_target))\
    .shuffle(buffer_size=10000)\
    .padded_batch(model_config.Hyperparams.batch_size,
                  (([None], []), ([None], [])),
                  ((conf.PAD_ID, 0), (conf.PAD_ID, 0)))\
    .prefetch(1)

#development dataset
dev_source = _file_to_ID_seq(model_config.Config.source_dev_tok,
                             model_config.Config.vocab_source)
dev_target = _file_to_ID_seq(model_config.Config.target_dev_tok,
                             model_config.Config.vocab_target)
dev_data = tf.data.Dataset.zip((dev_source, dev_target))\
    .padded_batch(model_config.Hyperparams.batch_size,
                  (([None], []), ([None], [])),
                  ((conf.PAD_ID, 0), (conf.PAD_ID, 0)))\
    .prefetch(1)

# iterator and handle
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle,
                                               train_data.output_types,
                                               train_data.output_shapes)
(x, x_lengths), (y, y_lengths) = iterator.get_next()
train_iterator = train_data.make_initializable_iterator()
dev_iterator = dev_data.make_initializable_iterator()

"""graph for train""" 
# encoder
encoder = Encoder(x, x_lengths,  hp, is_training=True)

# decoder
# add <S> (id:2) to the head and remove the last position of y, resulting a tensor with the same as y.
dec_inputs = tf.concat([tf.ones_like(y[:, :1]) * conf.SOS_ID, y[:,:-1]], axis=-1) 
decoder = Decoder(dec_inputs, y_lengths,  encoder.outputs, encoder.enc_mask, hp, True)

#------------------ loss and optimizer---------------------
#mask on output sequence
is_target = decoder.dec_mask
zero_pad = tf.zeros_like(is_target, dtype=tf.float32)
is_target_int = tf.cast(tf.not_equal(y, 0), dtype=tf.float32) # 1 for target positions and 0 for paddings

#number of tokens in the batch
n_batch_tokens = tf.cast(tf.reduce_sum(decoder.lengths), dtype=tf.float32) 

# token with maximum likelihood
ml_pred = tf.cast(tf.argmax(decoder.logits, axis=-1), tf.int32) #[N, max_seq_len+1]

#accuracy
acc = tf.reduce_sum(
    tf.where(is_target, tf.cast(tf.equal(y, ml_pred), dtype=tf.float32), zero_pad)
) / n_batch_tokens

#loss
labels_smoothed = label_smoothing(tf.one_hot(y, depth=hp.vocab_size))
loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=decoder.logits,
    labels=labels_smoothed
    )
batch_loss = tf.reduce_sum(tf.where(is_target, loss, zero_pad))
mean_loss = batch_loss / n_batch_tokens

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
train_op = optimizer.minimize(mean_loss)

""" make directories """
summary_dir = conf.logdir + '/summary'
checkpoint_dir = conf.logdir + '/checkpoint'
for p in [conf.logdir, summary_dir, checkpoint_dir]:
    if not os.path.exists(p):
        os.mkdir(p)

# saver
saver = tf.train.Saver()

# summary
train_summary_op = tf.summary.merge([
    tf.summary.scalar('accuracy', acc),
    tf.summary.scalar('mean_loss', mean_loss)
    ])
summary_writer = tf.summary.FileWriter(summary_dir)

#Inferencer
inferencer = inference.Inference()

#session config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True

table_initializer = tf.tables_initializer()
with tf.Session(config=config) as sess:
    #debug
#    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    print("session start")
    # restore or initialize variables
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        saver.restore(sess, latest_checkpoint)
    else:
        sess.run(tf.global_variables_initializer())

    #initialize tables
    print("initializing tables")
    sess.run(table_initializer)

    # getting data iterator handles         
    train_handle, dev_handle = sess.run([train_iterator.string_handle(), dev_iterator.string_handle()])

    # development stats calculator
    def calc_dev():
        result = []
        sess.run(dev_iterator.initializer)
        while True:
            try:
                result.append(sess.run([n_batch_tokens,
                                        batch_loss,
                                        acc],
                                        feed_dict={handle: dev_handle}))
            except tf.errors.OutOfRangeError:
                break
        result = np.sum(result, axis=0)
        return result[1]/result[0], result[2]/result[0]
        
    def custom_summary(tag_value):
        return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value])

    # min dev_loss
    min_dev_loss = 2**30
    no_improve_count = 0

    # global step
    global_step = 0
    epoch = 0

    while True:
        # calculate accuracy over development data
        sys.stdout.write("------------ calculating loss and acc for dev data\r")
        dev_loss, dev_accuracy = calc_dev()

        #BLEU
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            sys.stdout.write("------------ evaluating with BLEU\r")
            bleu = inferencer.BLEU_evaluation_with_test_data(1, latest_checkpoint)
        else:
            bleu = 0

        # print log
        print("{{'epoch': {}, 'dev_acc': {}, 'dev_loss': {}, 'BLEU': {}, 'time': '{}'}},"
            .format(epoch,
                    dev_accuracy,
                    dev_loss,
                    bleu,
                    datetime.datetime.now()))

        # finish training when dev_loss hasnt improved in 3 contiguous epochs
        if min_dev_loss > dev_loss:
            no_improve_count = 0
            min_dev_loss = dev_loss
        else:
            no_improve_count += 1
            if no_improve_count >= 5:
                break

        # training
        #initialize train dataset iterator
        sess.run(train_iterator.initializer)

        #epoch-local step counter and timestamp
        local_step = 0
        local_time_start = time.time()

        #training loop for one epoch
        while True:
            try:
                if(global_step % 100 == 0):
                    # train and get summary on training data
                    train_summary, _ = sess.run(
                        [train_summary_op, train_op],
                        feed_dict={handle: train_handle})

                    #write summary on train data
                    summary_writer.add_summary(train_summary, global_step)

                    #write summary on dev data
                    dev_loss, dev_accuracy = calc_dev()
                    summary_writer.add_summary(
                        custom_summary([['dev_loss', dev_loss],
                                        ['dev_accuracy', dev_accuracy]]),
                        global_step)

                    #display training info
                    sys.stdout.write("local step:{}, local time elapsed:{}"\
                        ", time per step:{}, global step:{}\r".format(
                        local_step,
                        time.time() - local_time_start,
                        (time.time()-local_time_start)/local_step if local_step>0 else "-",
                        global_step
                    ))
                else:
                    sess.run(train_op, feed_dict={handle: train_handle})

                global_step = global_step + 1
                local_step = local_step + 1
            except tf.errors.OutOfRangeError:
                global_step = global_step + 1
                local_step = local_step + 1
                break

        #save parameters after each epoch
        sys.stdout.write("------------ saving parameters of epoch {}\r".format(epoch))
        saver.save(sess, checkpoint_dir + '/' + conf.model_name, global_step=global_step)
        epoch = epoch + 1
