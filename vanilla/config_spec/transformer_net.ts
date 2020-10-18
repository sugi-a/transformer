namespace TransformerBasic {
    export type config = {
        'd_model': bigint,
        'maxlen': bigint,
        'ff_size': bigint,
        'dropout_rate': number,
        'n_enc_blocks': bigint,
        'n_dec_blocks': bigint,
        'use_pos_enc': boolean,
        'use_pos_emb': boolean,
        'share_enc_dec_embedding': boolean,
        'vocab_size': bigint,
    } & (
        {
            'use_rel_pos': true,
            'rel_pos_max_dist': bigint,
            'rel_pos_unique_per_head': boolean
        } | {
            'use_rel_pos': false
            'rel_pos_max_dist': null,
            'rel_pos_unique_per_head': null
        }
    );

}