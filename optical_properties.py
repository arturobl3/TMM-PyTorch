from typing import List, Tuple, Literal
from T_matrix import T_matrix
import torch 

class OpticalProperties:
    def __init__(self,
                Tm_s: T_matrix,
                Tm_p: T_matrix,
                n_env:torch.Tensor,
                n_subs: torch.Tensor,
                nx:torch.Tensor    
    ) -> None:
        self.Tm_s = Tm_s
        self.Tm_p = Tm_p
        self.n_env = n_env
        self.n_subs = n_subs
        self.nx = nx
        
    def reflection(self):
      
        r_s = self.Tm_s[:, :, 1, 0]/self.Tm_s[:, :, 0, 0]
        r_p = self.Tm_p[:, :, 1, 0]/self.Tm_p[:, :, 0, 0]

        return torch.abs(r_s)**2, torch.abs(r_p)**2
        

    def transmission(self):

        n1z = torch.sqrt(self.n_env[:,None]**2 - self.nx**2)
        n2z = torch.sqrt(self.n_subs[:,None]**2 - self.nx**2)

        t_s = 1/self.Tm_s[:, :, 0, 0] 
        t_p = 1/self.Tm_p[:, :, 0, 0] 

        T_s = torch.abs(t_s)**2 * torch.real(n2z/n1z)
        T_p = torch.abs(t_p)**2 * torch.real(n2z/n1z)

        return T_s, T_p


    