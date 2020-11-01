type Int = number;
type Float = number;

export type Config = {
    size: Int,
    PAD_ID: Int,
    SOS_ID: Int,
    EOS_ID: Int,
    UNK_ID: Int,
    vocab_size: Int,
    dict: string
};