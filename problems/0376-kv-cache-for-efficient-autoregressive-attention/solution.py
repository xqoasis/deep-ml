import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    def online_sft(m, l, o, v, s):
        m_new = max(m, s)
        alpha = np.exp(m - m_new)
        p = np.exp(s - m_new)
        l = l * alpha + p
        o = o * alpha + p*v

        return m_new, l, o

    # got cache already
    k_new = x_new @ W_K #(1, d_k)
    v_new = x_new @ W_V
    q_new = x_new @ W_Q
    d_k = W_K.shape[1]
    d_v = W_V.shape[1]

    # 第一步空 cache 升成二维
    if cache is None or cache[0] is None:
        K_cache = k_new[np.newaxis, :]
        V_cache = v_new[np.newaxis, :]
    else:
        K_cache = np.vstack([cache[0], k_new])
        V_cache = np.vstack([cache[1], v_new])

    context_len= K_cache.shape[0]

    # TODO 3：算 scores = Q_new @ K.T / sqrt(d_k)，做稳定 softmax（减最大值），再乘 V 得 output
    attn_score = (K_cache @ q_new) / np.sqrt(d_k)
    # calc attn_step
    m = -np.inf
    l = 0.0
    o = np.zeros(d_v)

    for i in range(context_len):
        m, l, o = online_sft(m, l, o, V_cache[i], attn_score[i])
    return o / l, (K_cache, V_cache)  

