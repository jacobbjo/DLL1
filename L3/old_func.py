def batch_norm_back_pass_prev_try(djdshat, s, m, v):
    vb = v + 0.0000001  # adding small epsilon to avoid zero division
    Vb_12 = np.diag((np.power(vb, -1/2)).reshape(-1))
    Vb_32 = np.diag((np.power(vb, -3/2)).reshape(-1))

    n = s.shape[1]

    djdvb = np.zeros((1, s.shape[0]))
    djdmy = np.zeros((1, s.shape[0]))

    for i in range(n):
        score_diag = np.diag((s[:, i].reshape(-1, 1) - m).reshape(-1))
        temp_prod = np.matmul(djdshat[i, :].reshape(1, -1), Vb_32)
        djdvb += np.matmul(temp_prod, score_diag)

        djdmy += np.matmul(s[:, i].reshape(1, -1), Vb_12)

    djdvb *= (-1/2)
    djdmy *= (-1)

    djds = np.zeros(djdshat.shape)

    for i in range(n):
        score_diag = np.diag((s[:, i].reshape(-1, 1) - m).reshape(-1))
        part_1 = np.matmul(djdshat[i, :].reshape(1, -1), Vb_12)
        part_2 = (2/n) * np.matmul(djdvb, score_diag)
        part_3 = djdmy * (1/n)

        djds[i, :] = part_1 + part_2 + part_3

    return djds