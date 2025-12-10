"""
Implementation of S4 (efficient version)
"""
import torch 
import numpy as np 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as f 

def _hippo_matrix(N):
    """
    hippo_matrix
    n > k: -sqrt(2n + 1) sqrt(2k + 1)
    n = k: -(n + 1)
    n < k: 0
    """

    hippo_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i > j:
                hippo_mat[i,j] = np.sqrt((2 * i + 1) * (2 * j + 1))
            elif i == j:
                hippo_mat[i, j] = i + 1
    hippo_mat = -hippo_mat 
    return hippo_mat

def _make_hippo_nplr(N):
    """
    pp.T = 1/2 sqrt(2n + 1) sqrt(2k + 1)
    """

    hippo_mat = _hippo_matrix(N)
    p = np.sqrt(np.arange(N) + 0.5) 
    nplr = hippo_mat + np.outer(p,p.conj()) # A + pp.T = -1/2 I + S: S + S.T = 0

    _lambda, V = np.linalg.eig(nplr) # eigen values may be complex

    ### sanity check
    # check = V @ (np.diag(_lambda) @ (V.conj().T))
    # print(check.real)
    # print(nplr)

    P = V.conj().T @ p # V*P
    B = V.conj().T @ (np.sqrt(2) * p)
    # A' = np.diag(_lambda) - np.outer(p, p.conj().T)
    return {
        "lambda": _lambda,
        "P": P,
        "B": B
    }
    
class S4Compute():
    def _discretize(self, delta, _lambda, P, B):  # assuming delta is a number 
        I = torch.eye(B.shape[0])
        A0 = (2 / delta) * I + torch.diag(_lambda) - torch.outer(P, P.conj()) # 2/delta I + lambda - PQ*
        D = torch.diag(1/((2 / delta) - _lambda)) # (2 / delta - lambda)^-1
        it1 = D @ (torch.outer(P, P.conj())) @ D # intermediate term DP Q*D
        A1 = D - it1 / (1 + torch.dot(P.conj(), D @ P))
        Ab = A1 @ A0 
        Bb = 2 * A1 @ B 
        return Ab, Bb
    
    def _kernel_compute(self, z, delta, _lambda, C, B, P, L):
        def k_z(z, delta, _lambda, C, B, P): # this needs to be evaluated at roots of unity
            R = lambda z, delta, _lambda: torch.diag(1 / (((2 * (1 - z)) / (delta * (1 + z))) - _lambda))
            coef = 2 / (1 + z)
            R_z = R(z, delta, _lambda)
            it1 = R_z @ (torch.outer(P, P.conj())) @ R_z # intermediate term RP Q*R
            woodbury_inv = R_z - it1 / (1 + torch.dot(P.conj(), (R_z @ P)))
            kernel_z = coef * C @ woodbury_inv @ B
        
        omegas = torch.arange(L).mul(2j * torch.pi / L).exp()
        kernel = k_z(omegas, delta, _lambda, C, B, P) # this can be more efficient via cauchy kernel
        return torch.fft.ifft(kernel, dim = -1)
    
    def _kernel_compute_cauchy(self, z, delta, _lambda, C, B, P, L):
        # using cauchy kernel to compute DPLR kernel
        pass 


class S4dplr(nn.Module):
    def __init__(self, N):
        super().__init__()
        s4_inits = _make_hippo_nplr(N)



    def forward(self, x, mode = "conv"):
        pass 



if __name__ == "__main__":
    inits = _make_hippo_nplr(5)
    print(inits)
