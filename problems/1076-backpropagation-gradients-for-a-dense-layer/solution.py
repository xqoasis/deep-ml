import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    # a_prev   (K,)      前一层激活，K 个神经元
    # W        (J, K)    当前层权重
    # b        (J,)
    # z = W @ a_prev + b     (J,K) @ (K,) = (J,)   ✅
    # a = sigmoid(z)         (J,)
    # y        (J,)
    # C = 标量

    # forward
    z = W @ a_prev + b # J, 
    a = 1.0 / (1.0 + np.exp(-z)) # by element, J
    # C = sum_j (a_j - y_j)^2 -> C 导数

    # gradiant start
    # a 导数是 a * (1-a)
    dC_da = 2.0 * (a - y) # (J,)
    da_dz = a * (1.0-a) #(J, )
    dz = dC_da * da_dz #(J, )

    dC_dW = np.outer(dz, a_prev)  # (shape J x K), dz @ a_prev.T，numpy 里，一维数组的 .T 什么都不做
    dC_db = dz # (shape J)
    dC_da_prev = W.T @  dz #(shape K) #

    return {
        'dW': dC_dW.tolist(),
        'db': dC_db.tolist(),
        'da_prev': dC_da_prev.tolist(),
    }

