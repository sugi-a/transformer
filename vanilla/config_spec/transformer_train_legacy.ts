/**
 * This is the definition of the JSON format configuration of
 * `TrainMultiGPULegacy`.
 * The definition is written in Typescript format.
 */

namespace TransformerTrainLegacy {
    type lenSmoothConfig_ = {
            'method': 'segsort',
            'segsize': bigint
        };
    type LenSmoothConfig = lenSmoothConfig_ & {'post_shuf_buf_size': bigint};

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
        }
    };
}