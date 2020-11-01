type Int = number;
type Float = number;

type RelPos = {
        use_rel_pos: true,
        rel_pos_max_dist: Int,
        rel_pos_unique_per_head: boolean
    } | {
        use_rel_pos: false
        rel_pos_max_dist: null,
        rel_pos_unique_per_head: null
    };

export type Config = {
    maxlen: Int,
    n_blocks: Int,
    n_heads: Int,
    d_model: Int,
    ff_size: Int,
    dropout_rate: Float,
    use_pos_enc: boolean,
    use_pos_emb: boolean,
    vocab_size: Int
} & RelPos;