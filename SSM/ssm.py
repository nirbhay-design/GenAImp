"""
Implementation of S4 (efficient version)
"""
import torch 
import numpy as np 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as f 


class HippoInitialize(object):
    def __init__(self, N):
        self.N = N 
        
    def _hippo_matrix(self):
        """
        hippo_matrix
        n > k: -sqrt(2n + 1) sqrt(2k + 1)
        n = k: -(n + 1)
        n < k: 0
        """
        N = self.N 

        hippo_mat = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i > j:
                    hippo_mat[i,j] = np.sqrt((2 * i + 1) * (2 * j + 1))
                elif i == j:
                    hippo_mat[i, j] = i + 1
        hippo_mat = -hippo_mat 
        return hippo_mat

    def _make_hippo_nplr(self):
        """
        p = 1/2 sqrt(2n + 1) sqrt(2k + 1)
        """

        N = self.N 
        hippo_mat = self._hippo_matrix()
        p = np.sqrt(np.arange(N) + 0.5) 
        nplr = hippo_mat + np.outer(p,p) # A + pp.T = -1/2 I + S: S + S.T = 0

        lambdas, V = np.linalg.eig(nplr) # eigen values may be complex

        ### sanity check
        # check = V @ (np.diag(lambdas) @ (V.conj().T))
        # print(check.real)
        # print(nplr)

        p = V.conj().T @ p # V^*P
        # A' = np.diag(lambdas) - np.outer(p, p.conj().T)
        return lambdas, p
    
class S4Compute():
    def _discretize(self, delta, lambdas, p, B):
        I = np.eye(B.shape[0])
        A0 = 0.5 * delta * I + torch.diag(lambdas) - torch.outer(p, p.conj()) # 2/delta I + lambda - PQ*
        D = torch.diag(1/(0.5 * delta - lambdas))
        A1 = D - (D @ p) @ (p.conj().T @ D) / (1 + p.conj().T @ D @ p)
        Ab = A1 @ A0 
        Bb = 2* A1 @ B 
        return Ab, Bb


class S4dplr(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.hippo_init = HippoInitialize(N)
        lambdas, p = self.hippo_init._make_hippo_nplr()


    def forward(self, x, mode = "conv"):
        pass 



if __name__ == "__main__":
    lambdas, p = HippoInitialize(5)._make_hippo_nplr()
    print(lambdas)
    print(p)
