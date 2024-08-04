import torch


def apply_coupling(q, qd, q_des, qd_des, kp, kd, tau_ff):
    # Create a Jacobian matrix and move it to the same device as input tensors
    J = torch.eye(q.shape[-1]).to(q.device)
    J[4, 3] = 1
    J[9, 8] = 1

    # Perform transformations using Jacobian
    q = torch.matmul(q, J.T)
    qd = torch.matmul(qd, J.T)
    q_des = torch.matmul(q_des, J.T)
    qd_des = torch.matmul(qd_des, J.T)

    # Inverse of the transpose of Jacobian
    J_inv_T = torch.inverse(J.T)

    # Compute feed-forward torques
    tau_ff = torch.matmul(J_inv_T, tau_ff.T).T

    # Compute kp and kd terms
    kp = torch.diagonal(
        torch.matmul(
            torch.matmul(J_inv_T, torch.diag_embed(kp, dim1=-2, dim2=-1)),
            J_inv_T.T
        ),
        dim1=-2, dim2=-1
    )

    kd = torch.diagonal(
        torch.matmul(
            torch.matmul(J_inv_T, torch.diag_embed(kd, dim1=-2, dim2=-1)),
            J_inv_T.T
        ),
        dim1=-2, dim2=-1
    )

    # Compute torques
    torques = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff
    torques = torch.matmul(torques, J)

    return torques