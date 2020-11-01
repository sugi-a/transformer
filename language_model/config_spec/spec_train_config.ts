type Int = number;
type Float = number;

type BatchSampling = {
        mode: 'random_sliding_window',
        window_size: Int,
        keep_remainder_larger_equal: Int,
        header: string
    } | {
        mode: 'sentence_sliding_window',
        window_size: Int
    }

export type Config = {
    batch: {
        sampling: BatchSampling,
        shuffle_buf_size: Int,
        batch_size: Int,
    },
    random_seed: Int,
    warm_up_steps: Int,
    label_smoothing: Float,
    max_step: Int,
    max_epoch: Int,
    summary_interval: Int, // in steps
    early_stop_patience: Int, // in epochs
    data: {
        train: string[],
        dev: string
    }
};