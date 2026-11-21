import numpy as np

def paged_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_table: list,
    context_len: int
) -> np.ndarray:
    """
    Perform scaled dot-product attention with paged KV cache.

    Args:
        query:       (num_heads, head_dim) query for a single token
        key_cache:   (num_physical_blocks, block_size, num_heads, head_dim)
        value_cache: (num_physical_blocks, block_size, num_heads, head_dim)
        block_table: list of physical block indices (logical -> physical)
        context_len: number of valid KV tokens in the sequence

    Returns:
        (num_heads, head_dim) attention output, rounded to 4 decimal places
    """

    def calc(q: np.ndarray, k: np.ndarray, v: np.ndarray):
        # k and v is shared by this head
        # 打分
        # 除以 √D 是把分数拉回到一个和 D 无关的量级上。（和len无关）
        # 一个点积要把 head_dim 个乘积加起来，head_dim 越大，加出来的数就越大。
        attn_score = k @ q/np.sqrt(len(q)) # 改：点积 + 缩放

        # softmax: 
        # w = exp(s − m) / Σ exp(s − m) → (L,)
        attn_score = attn_score - attn_score.max() # 防止 exp 溢出
        w = np.exp(attn_score)
        w = w / w.sum() # weight
        o = w @ v
        return o


    # block_table[logical_block_id] = physical_block_id
    num_heads, head_dim = query.shape
    block_size = key_cache.shape[1]
    to_calc_blocks_num = (context_len + block_size - 1) // block_size # 向上取整, 然后正好装满的时候 不会多算

    # handle key value shared by head
    keys = [[] for _ in range(num_heads)]
    values = [[] for _ in range(num_heads)]

    # get q and k and v for each head. shape is [L, head dim]
    for lid in range(to_calc_blocks_num):
        bid = block_table[lid]
        for off in range(block_size):
            # get the row
            position = lid * block_size + off
            if position >= context_len:
                # 没塞满的blocks
                break
            # for each head
            for head in range(num_heads):
                # get k and v for diff head
                k = key_cache[bid, off, head]
                v = value_cache[bid, off, head]
                keys[head].append(k)
                values[head].append(v)


    result = np.zeros((num_heads, head_dim))
    for head in range(num_heads):
        q = query[head]
        k = keys[head]
        v = values[head]
        out = calc(q, k, v)
        result[head] = out
    
    return np.round(result,4)

