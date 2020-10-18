/**
 * This is the definition of the JSON format configuration of
 * `TrainMultiGPULegacy`.
 * The definition is written in Typescript format.
 */

namespace TrainLegacy {
    type lenSmoothConfig_ = {
            'method': 'segsort',
            'segsize': bigint
        };
    type LenSmoothConfig = lenSmoothConfig_ & {'post_shuf_buf_size': bigint};

    type CkptConfig = ({
            'interval_type': 'constant',
            'interval': bigint,
        } | {
            'interval_type': 'exponential',
            'c1': bigint,
            'r': number
        })

    export type config = {
        'batch': {
            'constraint': ('capacity' | 'size')
            'size': bigint,
            'shuffle_buffer_size': bigint | null,
            'length_smoothing': null | LenSmoothConfig
        },
        'random_seed': number,
        'warm_up_step': bigint,
        'label_smoothing': number,
        'max_step': bigint,
        'max_epoch': bigint,
        'early_stopping_patience': bigint, // in epochs
        'early_stopping_criterion': 'loss' | 'bleu',
        'summary_interval': bigint, // in steps
        'data': {
            'source_train': string[],
            'target_train': string[],
            'source_dev': string,
            'target_dev': string
        },
        'vocab': {
            "PAD_ID": bigint,
            "SOS_ID": bigint,
            "EOS_ID": bigint,
            "UNK_ID": bigint,
            "source_dict": string,
            "target_dict": string,
        }
    };
}