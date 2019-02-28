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
parser.add_argument('--n_gpus', default=1, type=int,
    help="Number of GPUs.")
args = parser.parse_args()

sys.path.insert(0, args.model_dir)
import model_config
print("model_config has been loaded from {}".format(model_config.__file__))

from model_config import Hyperparams as hp
from model_config import Config as conf
import inference

def loss_fn(x, x_lengths, y, y_lengths, is_training=True):
    """computes loss for the given source and target sentences.
    Args:
        x,y: sequences of IDs representing the source and target sentences respectively. 
        x_lengths, y_lengths: lengths of x and y respectively.
        Each sequence must have an EOS ID at the end.
    Returns:
        loss, accuracy, n_tokens_in_batch
        All of them are scalar Tensors."""

    encoder = Encoder(x, x_lengths,  hp, is_training=is_training)
    
    # decoder
    # add <S> (id:2) to the head and remove the last position of y, resulting a tensor with the same as y.
    dec_inputs = tf.concat([tf.ones_like(y[:, :1]) * conf.SOS_ID, y[:,:-1]], axis=-1) 
    decoder = Decoder(dec_inputs,
                      y_lengths,
                      encoder.outputs,
                      encoder.enc_mask,
                      hp,
                      is_training,
                      encoder.embedding_weight if hp.share_embedding else None)
    
    #mask on output sequence
    is_target = tf.sequence_mask(y_lengths, tf.shape(y)[1])
    zero_pad = tf.zeros_like(is_target, dtype=tf.float32)
    
    #number of tokens in the batch
    n_batch_tokens = tf.cast(tf.reduce_sum(y_lengths), dtype=tf.float32) 
    
    # tokens with maximum likelihood
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

    return mean_loss, acc, n_batch_tokens

def make_optimization(loss_fn, input_fn, optimizer):
    """Makes optimization operation with a single GPU
    Returns:
        train_op, loss, accuracy"""
    x, x_lengths, y, y_lengths = input_fn()
    loss, acc, _n_tok = loss_fn(x, x_lengths, y, y_lengths)
    global_step = tf.train.get_global_step()
    assert global_step is not None

    return optimizer.minimize(loss, global_step), loss, acc

def make_parallel_optimization(loss_fn, input_fn, optimizer):
    """Makes optimization operation with args.n_gpus GPUs
    Returns:
        train_op, loss, accuracy"""

    tower_grads = []
    tower_losses = []
    tower_accuracy = []
    tower_n_toks = []

    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i in range(args.n_gpus):
            x, x_lengths, y, y_lengths = input_fn()

            with tf.device('/gpu:{}'.format(i)), tf.name_scope('tower_{}'.format(i)):
                loss, acc, n_toks = loss_fn(x, x_lengths, y, y_lengths)
                tower_losses.append(loss)
                tower_accuracy.append(acc)
                tower_n_toks.append(n_toks)

                with tf.name_scope('compute_gradient'):
                    # grads is list of (grad_k, variable_k)
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

            outer_scope.reuse_variables()

    avg_grads = []
    with tf.name_scope("averaging"):
        # total number of tokens in the batch
        total_toks = tf.add_n(tower_n_toks)
        for grads_vars in zip(*tower_grads):
            grads = [g_v[0] * tower_n_toks[i] for i, g_v in enumerate(grads_vars)]
            avg_grad = tf.add_n(grads) / total_toks
            avg_grads.append((avg_grad, grads_vars[0][1]))

        avg_loss = tf.add_n([l*w for l,w in zip(tower_losses, tower_n_toks)]) / total_toks 
        avg_acc = tf.add_n([a*w for a,w in zip(tower_accuracy, tower_n_toks)]) / total_toks

    with tf.name_scope("apply_gradients"):
        global_step = tf.train.get_global_step()
        train_op = optimizer.apply_gradients(avg_grads, global_step)

    return train_op, avg_loss, avg_acc


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

def train():

    # arguments check
    assert hp.batch_size % args.n_gpus == 0
    from tensorflow.python.client import device_lib
    assert len([x for x in device_lib.list_local_devices() if x.device_type=='GPU']) >= args.n_gpus

#train dataset
    train_data = (_create_dataset(
                    conf.source_train_tok,
                    conf.target_train_tok,
                    conf.vocab_source,
                    conf.vocab_target,
                    2000*1000)
        .filter(lambda x,y: tf.logical_and(tf.greater(hp.maxlen, x[1]),
                                         tf.greater(hp.maxlen, y[1])))
        .padded_batch(model_config.Hyperparams.batch_size // args.n_gpus,
                      (([None], []), ([None], [])),
                      ((conf.PAD_ID, 0), (conf.PAD_ID, 0)))
        .prefetch(args.n_gpus * 2))

#development dataset
    dev_data = _create_dataset(
                    conf.source_dev_tok,
                    conf.target_dev_tok,
                    conf.vocab_source,
                    conf.vocab_target
                )\
        .padded_batch(model_config.Hyperparams.batch_size // args.n_gpus,
                      (([None], []), ([None], [])),
                      ((conf.PAD_ID, 0), (conf.PAD_ID, 0)))\
        .prefetch(1)

    train_iterator = train_data.make_initializable_iterator()
    dev_iterator = dev_data.make_initializable_iterator()

    def input_fn(iterator=train_iterator):
        with tf.device(None):
            (x, x_lengths), (y, y_lengths) = iterator.get_next()
        
        return x, x_lengths, y, y_lengths


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
    train_op, train_loss, train_accuracy = (
        make_optimization(loss_fn, input_fn, optimizer) if args.n_gpus <= 1
        else make_parallel_optimization(loss_fn, input_fn, optimizer) 
    )

    # development data
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        dev_loss, dev_acc, dev_n_toks = loss_fn(*input_fn(dev_iterator), is_training=False)

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
        tf.summary.scalar('accuracy', train_accuracy),
        tf.summary.scalar('mean_loss', train_loss),
        tf.summary.scalar('learning_rate', lr)
        ])
    summary_writer = tf.summary.FileWriter(summary_dir)

#Inferencer
    inferencer = inference.Inference()

#mutable-nodes initializers
    table_initializer = tf.tables_initializer()
    variable_initializer = tf.global_variables_initializer()

#close the graph
    tf.get_default_graph().finalize()

#session config
    config = tf.ConfigProto()
   # config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.allow_soft_placement = True

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

        # development stats calculator
        def calc_dev():
            result = []
            sess.run(dev_iterator.initializer)
            while True:
                try:
                    result.append(sess.run([dev_loss, dev_acc, dev_n_toks]))
                except tf.errors.OutOfRangeError:
                    break
            result = np.array(result)
            # weighted average with the number of tokens as weight
            result = np.average(result, 0, result[:, 2])
            return result[0], result[1]
            
        def custom_summary(tag_value):
            """
            Args:
                tag_value: list of (tag, value)"""
            return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value])

        # max bleu
        max_bleu = -100
        no_improve_count = 0

        # global step
        global_step = sess.run(global_step_var)
        epoch = 0

        while True:
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
            print("{{'epoch': {}, 'BLEU': {}, 'time': '{}'}},"
                .format(epoch,
                        bleu,
                        datetime.datetime.now()))

            # finish training when the bleu score has not been improved for 5 contiguous epochs
            if bleu > max_bleu:
                no_improve_count = 0
                max_bleu = bleu
            else:
                no_improve_count += 1
                if no_improve_count >= 4:
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
                            [train_summary_op, global_step_var, train_op])

                        #write summary on train data
                        summary_writer.add_summary(train_summary, global_step - 1)

                        #write summary on dev data
                        _dev_loss, _dev_acc = calc_dev()
                        summary_writer.add_summary(
                            custom_summary([['dev_loss', _dev_loss],
                                            ['dev_accuracy', _dev_acc]]),
                            global_step - 1)

                        #display training info
                        sys.stdout.write("local step:{}, local time:{}"\
                            ", t/s:{}, global step:{} \r".format(
                            local_step,
                            time.time() - local_time_start,
                            (time.time()-local_time_start)/local_step if local_step>0 else "-",
                            global_step
                        ))
                    else:
                        global_step, _ = sess.run([global_step_var, train_op])

                    local_step = local_step + 1
                except tf.errors.OutOfRangeError:
                    break

            #save parameters after each epoch
            sys.stdout.write("------------ saving parameters of epoch {}\r".format(epoch))
            saver.save(sess, checkpoint_dir + '/' + conf.model_name, global_step=global_step)
            epoch = epoch + 1

if __name__ == "__main__":
    train()
