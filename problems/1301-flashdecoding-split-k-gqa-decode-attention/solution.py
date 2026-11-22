import numpy as np

def flash_decoding(q, k, v, n_splits):
    """
    FlashDecoding attention for one decode step with GQA.

    q: (batch, n_q_heads, head_dim)
    k, v: (batch, seq, n_kv_heads, head_dim)
    n_splits: int
    Returns: (batch, n_q_heads, head_dim)
    """
    def attn_step(m, l, o, s, v):
        m_new = np.maximum(m, np.max(s))
        alpha = np.exp(m - m_new)
        p = np.exp(s - m_new)

        l = l*alpha + np.sum(p)
        o = o*alpha + p@v

        return m_new, l, o
    
    b_size = q.shape[0]
    q_head = q.shape[1]
    head_dim = q.shape[2]
    context_len = k.shape[1]
    kv_head = k.shape[2]
    q_shared_kv_size = q_head // kv_head
    # 向上取整
    split_size = (context_len + n_splits - 1) // n_splits
    res = np.zeros_like(q, dtype = np.float32)

    for b in range(b_size):
        for q_head_id in range(q_head):
            kv_head_id = q_head_id // q_shared_kv_size
            cur_q = q[b, q_head_id]
            cur_k = k[b, :, kv_head_id, :]
            cur_v = v[b, :, kv_head_id, :]

            # init 
            m_list, l_list, o_list = [], [], []
            for i in range(n_splits):
                start = i * split_size
                if start >= context_len:
                    continue

                # calc attention score
                split_k = cur_k[start: start+split_size]
                split_v = cur_v[start: start+split_size]
                split_attn_score = split_k @ cur_q/np.sqrt(head_dim)
                split_m, split_l, split_o = attn_step(-np.inf, 0.0, np.zeros(head_dim), split_attn_score, split_v)

                m_list.append(split_m)
                l_list.append(split_l)
                o_list.append(split_o)

            
            m_max = max(m_list)
            l_g = 0.0 # batch, 1
            o_g = np.zeros(head_dim) # batch, head_dim
            # # 改：按实际非空段数合并
            for i in range(len(m_list)):
                alpha = np.exp(m_list[i] - m_max)
                l_g += l_list[i]*alpha
                o_g += o_list[i]*alpha
            res[b, q_head_id] = o_g / l_g

    return res

