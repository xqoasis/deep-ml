import numpy as np

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                           block_size: int = 2) -> np.ndarray:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    def attn_step(m, o, l, s, v):
        # multiple query 完全并行 只需要考虑 k and v 的广播
        m_max = np.maximum(m, s.max(axis = 1, keepdims = True)) # 行最大 + argmax
        alpha = np.exp(m - m_max)
        p = np.exp(s - m_max)
        l = l * alpha + p.sum(axis = 1, keepdims=True)
        o = o * alpha + p@v
        return m_max, o, l


    num_queries = Q.shape[0]
    head_dim = Q.shape[1]
    context_len = K.shape[0]
    # res shape the same as Q
    res = np.zeros_like(Q, dtype = np.float32)

    for i in range(0, num_queries, block_size):
        Q_i = Q[i:i+block_size]
        br = Q_i.shape[0] # 兼容最后一块不满

        # init 
        M = np.full((br, 1), -np.inf) # shape is block_size (query size)
        L = np.zeros((br, 1))
        O = np.zeros((br, head_dim))

        for j in range(0, context_len, block_size):
            Kj = K[j:j+block_size]
            Vj = V[j:j+block_size]
            # cal attn score for all queries
            Sj = Q_i @ Kj.T/np.sqrt(head_dim) 
            #Q_i @ Kj.T 得 [br, bc]

            # update result
            M, O, L = attn_step(M, O, L, Sj, Vj)

        OUT = O/L
        res[i:i+block_size] = OUT
    return res
