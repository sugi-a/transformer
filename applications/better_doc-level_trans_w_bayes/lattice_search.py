import sys
from itertools import chain
import numpy as np

def pad_list2D(lst2D):
    L = max(map(len, lst2D))
    return [row + [0] * (L - len(row)) for row in lst2D]


def search(
        hypos,
        forward_scores,
        backward_scores,
        fn_cond_lm,
        l1, l2, l3, 
        B, n_ctx):
    """
    Args:
        hypos: target sequences in IDs. Array<IDs, [N, K]>
        forward_scores: Array<f[N, K], float>
        backward_scores: Array<[N, K], float>
        fn_cond_lm: (
                context: <[B, L_ctx], int>,
                main: <[B, L_main], int>
            )=><[B], float>
            context does not need to be offsetted
        l1-l3: lambda1, ... , lambda3
        B: Beam size

        IDs = list<int>
    Returns:
        int[N]
    """
    N = len(hypos)
    K = len(hypos[0])

    def cond_lm_(self, ctx, cur_hypos):
        """
        Args:
            ctx: list<IDs, B>
            cur_hypos: list<IDs, K>
        Returns:
            Array<[B * K], float>
        """
        # list<IDs, B * K>
        ctx_tiled = list(chain.from_iterable((seq,)*K for seq in ctx))
        hypos_tiled = cur_hypos * B
        
        ctx_tiled = np.array(pad_list2D(ctx_tiled))
        hypos_tiled = np.array(pad_list2D(hypos_tiled))

        scores = fn_cond_lm(ctx_tiled, hypos_tiled)
        return scores.numpy()

    # Beam search
    # int[B, N]
    paths = np.zeros([B, N], int)
    # float[B]
    path_scores = np.zeros(B, float)

    for i in range(N):
        # log p_LM(y_i | y_{<i})
        ctx_start = max(0, i - n_ctx)
        # list<list<int>, B>
        ctx = [
            list(chain.from_iterable(
                hypos[n][paths[b][n]] for n in range(ctx_start, i)
            )) for b in range(B)]

        # [B * K]
        lm_score = cond_lm_(ctx, hypos[i])
        bt_score = np.tile(backward_scores[i], B)
        fw_score = np.tile(forward_scores[i], B)
        lengths = np.tile(np.array(len(s) for s in hypos[i]), B)
        prev_score = path_scores.repeat(K)

        score = (
            lm_score
            + l1 * fw_score
            + l2 * bt_score
            + l3 * lengths
            + prev_score)

        # Top B (sorted in ascending order)
        # 0 <= i < B * K
        indices = score.argsort()[-B:]

        # Choose the alive paths and scores
        paths = paths[indices // K]
        path_scores = score[indices]

        # Append new tokens
        paths[i] = indices % K


    # paths: [B, N] of IDs
    return paths[-1]

    