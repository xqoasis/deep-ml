import numpy as np

def manual_cross_entropy_backward(logits: np.ndarray, y: np.ndarray) -> dict:
    """
    Manually compute gradients through every intermediate of softmax + NLL loss.

    Args:
        logits: numpy array of shape (n, C)
        y:      numpy array of integer class indices, shape (n,)

    Returns:
        Dictionary with keys 'dlogprobs', 'dprobs', 'dcounts_sum_inv',
        'dcounts_sum', 'dcounts', 'dnorm_logits', 'dlogit_maxes', 'dlogits'.
    """
    n, C = logits.shape # n is the out feature, C is the sample number
    # foward
    logit_maxes = logits.max(axis = 1, keepdims = True) #(n, 1)
    norm_logits = logits - logit_maxes # (n, C)
    counts = np.exp(norm_logits) # (n, C)
    counts_sum = counts.sum(axis=1, keepdims = True) # (n, 1)
    counts_sum_inv = counts_sum ** -1 # (n, 1)
    probs = counts * counts_sum_inv # (n, C)

    # backwards
    # start is the loss
    # # ① loss = -mean(logprobs[range(n), y])
    dlogprobs = np.zeros_like(probs)
    dlogprobs[np.arange(n), y] = -1.0/n
    # ② logprobs = log(probs)
    dprobs = dlogprobs / probs
    # ③ probs = counts * counts_sum_inv   —— counts_sum_inv 被广播，要 sum
    dcounts_sum_inv = (dprobs * counts).sum(axis = 1, keepdims = True)
    # another 乘法
    dcounts = dprobs * counts_sum_inv
    # counts_sum_inv = counts_sum ** -1
    dcounts_sum = -dcounts_sum_inv * counts_sum ** (-2)
    # ⑥ counts_sum = counts.sum(axis=1) —— counts 的第二份贡献，和上面的加起来
    dcounts += np.ones_like(counts)*dcounts_sum
    #  ⑦ counts = exp(norm_logits)
    dnorm_logits = dcounts * np.exp(norm_logits)
    # norm_logits = logits - logits_max, 减法的两边都回退
    dlogit_maxes = (-dnorm_logits).sum(axis = 1, keepdims = True)
    dlogits = dnorm_logits.copy()

    # max 搞回去
    one_hot_max = np.zeros_like(logits, dtype = float)
    one_hot_max[np.arange(n), logits.argmax(axis = 1)] = 1.0
    dlogits += one_hot_max * dlogit_maxes

    return {
        'dlogprobs':       dlogprobs,
        'dprobs':          dprobs,
        'dcounts_sum_inv': dcounts_sum_inv,
        'dcounts_sum':     dcounts_sum,
        'dcounts':         dcounts,
        'dnorm_logits':    dnorm_logits,
        'dlogit_maxes':    dlogit_maxes,
        'dlogits':         dlogits,
    }




