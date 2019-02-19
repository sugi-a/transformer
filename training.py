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
parser.add_argument('--n_cpu_cores', default=8, type=int,
    help="Number of CPU cores this script can use when preprocessing dataset")
args = parser.parse_args()

sys.path.insert(0, args.model_dir)
import model_config
print("model_config has been loaded from {}".format(model_config.__file__))

from model_config import Hyperparams as hp
from model_config import Config as conf
import inference

# load dataset
def _create_dataset(source_file_name, target_file_name, source_vocab_file_name, target_vocab_file_name, shuffle_size=None):
    """load file into dataset"""
    tables = [tf.contrib.lookup.index_table_from_file(
                vocab_file_name,
                num_oov_buckets=0,
                default_value=conf.UNK_ID,
                key_column_index=0)
                for vocab_file_name in (source_vocab_file_name, target_vocab_file_name)]
    ncpu = args.n_cpu_cores
    source_dataset = tf.data.TextLineDataset(source_file_name)
    target_dataset = tf.data.TextLineDataset(target_file_name)
    dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
    if shuffle_size is not None:
        dataset = dataset.shuffle(shuffle_size)
    return (dataset
        .map(lambda s,t: tuple(tf.string_split([line]).values for line in [s,t]), ncpu)
        .map(lambda s,t: tuple(tf.cast(tables[i].lookup(tokens), tf.int32)
            for i, tokens in enumerate([s,t])), ncpu)
        .map(lambda s,t: tuple(tf.pad(seq, [[0, 1]], constant_values=conf.EOS_ID) for seq in [s,t]), ncpu)
        .map(lambda s,t: tuple((seq, tf.shape(seq)[0]) for seq in [s,t]), ncpu)
    )

#train dataset
train_data = (_create_dataset(
                conf.source_train_tok,
                conf.target_train_tok,
                conf.vocab_source,
                conf.vocab_target,
                1000*1000)
    .filter(lambda train,dev: tf.logical_and(tf.greater(hp.maxlen, train[1]),
                                     tf.greater(hp.maxlen, dev[1])))
    .padded_batch(model_config.Hyperparams.batch_size,
                  (([None], []), ([None], [])),
                  ((conf.PAD_ID, 0), (conf.PAD_ID, 0)))
    .prefetch(4))

#development dataset
dev_data = _create_dataset(
                conf.source_dev_tok,
                conf.target_dev_tok,
                conf.vocab_source,
                conf.vocab_target
            )\
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
decoder = Decoder(dec_inputs,
                  y_lengths,
                  encoder.outputs,
                  encoder.enc_mask,
                  hp,
                  True,
                  encoder.embedding_weight if hp.share_embedding else None)

#------------------ loss and optimizer---------------------
#mask on output sequence
is_target = decoder.dec_mask
zero_pad = tf.zeros_like(is_target, dtype=tf.float32)

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
# learning rate controller. argument "step" is a scalar Tensor
def get_learning_rate(step):
    WARM_UP_STEP = tf.cast(hp.warm_up_step, tf.float32) 
    step = tf.cast(step, tf.float32)
    rate = hp.embed_size ** (-0.5) * tf.minimum(tf.rsqrt(step),
                                                step * tf.pow(WARM_UP_STEP, -1.5))
    return rate

global_step_var = tf.train.get_or_create_global_step()
lr = get_learning_rate(global_step_var)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(mean_loss, global_step_var)

""" make directories """
summary_dir = conf.logdir + '/summary'
checkpoint_dir = conf.logdir + '/checkpoint'
for p in [conf.logdir, summary_dir, checkpoint_dir]:
    if not os.path.exists(p):
        os.mkdir(p)

# saver
saver = tf.train.Saver(max_to_keep=12)

# summary
train_summary_op = tf.summary.merge([
    tf.summary.scalar('accuracy', acc),
    tf.summary.scalar('mean_loss', mean_loss),
    tf.summary.scalar('learning_rate', lr)
    ])
summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())

#Inferencer
inferencer = inference.Inference()

#mutable-nodes initializers
table_initializer = tf.tables_initializer()
variable_initializer = tf.global_variables_initializer()

#close the graph
tf.get_default_graph().finalize()

#session config
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.log_device_placement = True

with tf.Session(config=config) as sess:
#    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    print("session start")
    # initialize and restore variables
    sess.run(variable_initializer)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        saver.restore(sess, latest_checkpoint)

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
        n_batches = len(result)
        result = np.sum(result, axis=0)
        return result[1]/result[0], result[2]/n_batches
        
    def custom_summary(tag_value):
        return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value])

    # min dev_loss
    min_dev_loss = 2**30
    no_improve_count = 0

    # global step
    global_step = sess.run(global_step_var)
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

        #epoch summary
        summary_writer.add_summary(custom_summary([['BLEU', bleu]]), global_step)

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
                if(global_step % 500 == 0):
                    # train and get summary on training data
                    train_summary, global_step, _ = sess.run(
                        [train_summary_op, global_step_var, train_op],
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
                    sys.stdout.write("local step:{}, local time:{}"\
                        ", t/s:{}, global step:{} \r".format(
                        local_step,
                        time.time() - local_time_start,
                        (time.time()-local_time_start)/local_step if local_step>0 else "-",
                        global_step
                    ))
                else:
                    global_step, _ = sess.run([global_step_var, train_op],
                                              feed_dict={handle: train_handle})

                local_step = local_step + 1
            except tf.errors.OutOfRangeError:
                break

        #save parameters after each epoch
        sys.stdout.write("------------ saving parameters of epoch {}\r".format(epoch))
        saver.save(sess, checkpoint_dir + '/' + conf.model_name, global_step=global_step)
        epoch = epoch + 1
