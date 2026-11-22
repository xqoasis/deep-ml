import math
import numpy as np


def attend_step(m, l, o, s, v):
    m_new = max(m, s)
    alpha = math.exp(m - m_new)
    p = math.exp(s - m_new)
    l_new = l*alpha + p
    o_new = o * alpha + p * v
    return (m_new, l_new, o_new)


def flash_attention(Q, K, V):
    num_query = Q.shape[0]
    context_len = K.shape[0]
    head_dim = Q.shape[1]

    res = np.zeros((num_query, head_dim))

    # 外层遍历每个 query
    for i in range(num_query):
        q = Q[i]
        # init m and l for each query
        m = -math.inf
        l = 0.0
        # 单个 query 额外空间是 O(d)（完全跟 L 无关）!!!!!!
        o = np.zeros(head_dim) # o 多乘一个 v 向量

        for ctx in range(context_len):
            k = K[ctx]
            v = V[ctx]
            # get attn score
            s = k@q / np.sqrt(head_dim)
            m, l, o = attend_step(m, l, o, s, v)
        res[i] = o/l
    return res.astype(np.float32)


    # 如果cuda 编程中 k and V read multiple lines per call
# above will become 向量


#     第 1 处是分数 s：从 1 个标量变成块数组 s_block [B_c]，所以找最高分要加一步块内求 max： m_new = max(m, np.max(s_block))（算完 m_new 还是 1 个标量）
# 第 2 处是票数 p：从 1 个标量变成块数组 p_block = np.exp(s_block - m_new)（[B_c]），所以进总票箱 l 时要先把块内求和： l_new = l * alpha + np.sum(p_block)（算完 l_new 还是 1 个标量）
# 至于 v，原本就是长度 [d] 的向量，分块后变成矩阵 V_block [B_c, d]，对应写成 p_block @ V_block（算完 o_new 还是长度 [d]）。


def reference_attention(Q, K, V):
    head_dim = Q.shape[1]
    attn_score = Q @ K.T / np.sqrt(head_dim)
    # 提示：axis=-1, keepdims=True 别漏，不然行最大值/行和会压成 1 维，除法广播会错。
    attn_score -= np.max(attn_score, axis=-1, keepdims=True)
    w = np.exp(attn_score)
    prob = w / np.sum(w, axis=-1, keepdims=True)
    return (prob @ V).astype(np.float32)


def score_matrix_bytes(N, elem_bytes=4):
    return N * N * elem_bytes
