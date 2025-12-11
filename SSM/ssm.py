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
    # A' = np.diag(_lambda) - np.outer(P, P.conj())
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


def _log_step_initilizer(delta, dmin = 1e-3, dmax = 0.1):
    """
    Keep delta between 0.001 to 0.1 and not let it overflow
    """
    lg_dmin = np.log(dmin)
    lg_dmax = np.log(dmax)
    scale = lg_dmax - lg_dmin 
    return delta * scale + lg_dmin 

class S4dplr(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        
        """
        d: input_dimension 
        h: hidden state dimension
        """
        s4_inits = _make_hippo_nplr(h)
        s4c = S4Compute()

        _lambda = torch.from_numpy(s4_inits["lambda"]).type(torch.complex64)
        P = torch.from_numpy(s4_inits["P"]).type(torch.complex64)
        B = torch.from_numpy(s4_inits["B"]).type(torch.complex64)

        ## Model Parameter Initialization
        self._lambda = nn.Parameter(torch.view_as_real(_lambda)) # (h,)
        self.P = nn.Parameter(torch.view_as_real(P)) #(h,)
        self.B = nn.Parameter(torch.view_as_real(B.unsqueeze(0).repeat((d, 1)))) # (d x h)
        self.C = nn.Parameter(torch.view_as_real(torch.rand(d, h, dtype=torch.complex64)))  # (d x h)
        self.D = nn.Parameter(torch.ones(1,1,d))
        self.log_delta = nn.Parameter(_log_step_initilizer(torch.rand(d))) # (d,)

    def _get_complex(self):
        _lambda = torch.view_as_complex(self._lambda)
        P = torch.view_as_complex(self.P)
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)
        return _lambda, P, B, C

    
    def forward(self, x, mode = "conv"):
        _lambda, P, B, C, D, delta = self._get_complex(), self.D, self.log_delta.exp() # all parameters in complex form



if __name__ == "__main__":
    s4dplr = S4dplr(32, 5)
    for name, value in s4dplr.named_parameters():
        print(f"{name}: {value.shape}")     
    
"""
def _discretize(self, delta, _lambda, P, B):
        # Discretize the continuous-time SSM (A, B) to discrete-time (Ab, Bb)
        # using the Bilinear Transform and Woodbury Identity.
        
        # Args:
        #     delta:   (D,)   Step sizes per channel
        #     _lambda: (H,)   Diagonal state matrix (Complex)
        #     P:       (H,)   NPLR Low-rank vector (Complex)
        #     B:       (D, H) Input matrix (Complex)
            
        # Returns:
        #     Ab: (D, H, H) Discrete Recurrence Matrix
        #     Bb: (D, H)    Discrete Input Matrix
        
        # 1. Shape Prep
        # We need to broadcast everything to (D, H) or (D, H, H)
        D_dim = delta.shape[0]
        H_dim = _lambda.shape[0]
        
        # delta: (D, 1) for broadcasting
        delta = delta.view(D_dim, 1)
        
        # 2. Compute the Inverse Term: A1 = (2/delta * I - A)^-1
        # A = diag(_lambda) - P @ P.conj()
        # Term to invert = (2/delta - _lambda) + P @ P.conj()
        
        # Diagonal part of the term to invert: D_row = (2/delta - _lambda)
        # Shape: (D, H) - Broadcasting delta against lambda
        D_row = (2.0 / delta) - _lambda
        
        # L = 1 / D_row (The diagonal inverse matrix if P was 0)
        L = 1.0 / D_row # Shape: (D, H)

        # Woodbury Identity: (D + UV)^-1 = D^-1 - D^-1 U (1 + V D^-1 U)^-1 V D^-1
        # Here U = P, V = P*
        # We need (1 + P* @ L @ P)
        
        # P_weighted: L @ P (Element-wise because L is diagonal)
        # Shape: (D, H)
        P_weighted = L * P # Broadcasts P (H,) against L (D, H)
        
        # Denominator scalar: 1 + P* @ L @ P
        # Sum over H dimension
        # Shape: (D, 1)
        woodbury_scalar = 1.0 + (P.conj() * P_weighted).sum(dim=-1, keepdim=True)
        
        # Now we construct the dense A1 matrix explicitly for recurrence
        # A1 = L_mat - (L P) @ (P* L) / scalar
        
        # L_mat: Diagonal matrix (D, H, H)
        L_mat = torch.diag_embed(L)
        
        # Low rank term: (D, H, 1) @ (D, 1, H) -> (D, H, H)
        low_rank_term = (P_weighted.unsqueeze(-1) @ (P_weighted.conj().unsqueeze(-2)))
        
        # A1: The calculated Inverse matrix
        # Shape: (D, H, H)
        A1 = L_mat - (low_rank_term / woodbury_scalar.unsqueeze(-1))
        
        # 3. Compute the Numerator Term: A0 = (2/delta * I + A)
        # A0 = diag(2/delta + _lambda) - P @ P.conj()
        
        # Diagonal part
        A0_diag = torch.diag_embed((2.0 / delta) + _lambda) # (D, H, H)
        
        # Low rank part (Same P P* for all D, just broadcasted)
        # Shape: (1, H, H) broadcasted to (D, H, H)
        A0_lowrank = torch.outer(P, P.conj()).unsqueeze(0)
        
        A0 = A0_diag - A0_lowrank # (D, H, H)
        
        # 4. Compute Discrete Matrices
        # Ab = A1 @ A0
        Ab = torch.bmm(A1, A0) # Batch matrix multiplication over D
        
        # Bb = 2 * A1 @ B
        # B is (D, H) -> unsqueeze to (D, H, 1) for matmul
        Bb = 2.0 * torch.bmm(A1, B.unsqueeze(-1)).squeeze(-1)
        
        return Ab, Bb
"""