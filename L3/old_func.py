import numpy.matlib as npm

def batch_norm_back_pass_tov(djdshat, s, m, v):
    n = s.shape[1]
    vb = v + 0.0000001  # adding small epsilon to avoid zero division
    Vb_12 = np.diag((np.power(vb, -1/2)).reshape(-1))
    Vb_32 = np.diag((np.power(vb, -3/2)).reshape(-1))

    score_m_vector = np.diag(s-m).reshape(1, -1)

    djdvb = (1/2) * np.sum(np.multiply(np.matmul(djdshat, Vb_32), score_m_vector), 0)
    djdmy = (-1) * np.sum(np.matmul(djdshat, Vb_12), 0)

    djdvb_mat = npm.repmat(djdvb, n, 1)
    djdmy_mat = npm.repmat(djdmy, n, 1)

    part_1 = np.matmul(djdshat, Vb_12)
    part_2 = (2/n) * np.multiply(djdvb_mat, score_m_vector)
    part_3 = djdmy_mat/n

    djds = (part_1 + part_2 + part_3)

    return djds