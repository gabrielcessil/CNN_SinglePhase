import numpy         as np
from   scipy.ndimage import distance_transform_edt
from   typing        import Union
import torch

def pressure_calculation_from_R(
    R:          int,
    L:          int,
    tau:        Union[float, int],
    Re:         float = 0.1,
    Dens:       float = 1.0,
    )->float:
    
    
    Visc            = (tau - 0.5) / 3.0
    dP              = (Re * 8.0 * (Visc ** 2) * L) / (Dens * (R ** 3))

    return dP

def force_calculation_from_R(
    R:              int,
    tau:            Union[float, int],
    Re:             float = 0.1,
    Dens:           float = 1.0,
) -> float:
    
    Visc            = (tau - 0.5) / 3.0
    Fx              = (Re * 8.0 * (Visc ** 2)) / (Dens * (R ** 3))
    
    return Fx

def pressure_calculation(
    matrix:     np.ndarray,
    tau:        Union[float, int],
    Re:         float = 0.1,
    Dens:       float = 1.0,
    is_edt:     bool  = False,
    )->float:
    
    L               = matrix.shape[0]
    
    dist_transform  = matrix if is_edt else distance_transform_edt(matrix>0)
    if dist_transform.size == 0 or np.max(dist_transform) == 0: return 0.0
    R               = np.max(dist_transform)

    return pressure_calculation_from_R(R=R, L=L, tau=tau, Re=Re, Dens=Dens)


def force_calculation(
    matrix:         np.ndarray,
    tau:            Union[float, int],
    Re:             float = 0.1,
    Dens:           float = 1.0,
    is_edt:         bool  = False,
) -> float:
    
    dist_transform  = matrix if is_edt else distance_transform_edt(matrix>0)
    if dist_transform.size == 0 or np.max(dist_transform) == 0: return 0.0
    R               = np.max(dist_transform)
    
    return force_calculation_from_R(R=R, tau=tau, Re=Re, Dens=Dens)


def tensor_denorm(out: torch.Tensor, 
                  inp: torch.Tensor):
    """
    out: (B, 4, Z, Y, X) -> Predições da rede
    inp: (B, 1, Z, Y, X) -> Input (EDT)
    """
    all_samples = [] # Usar lista é mais rápido que concatenar tensores no loop
    
    # Corrigido: range() é necessário para iterar sobre o batch
    for b in range(out.shape[0]):
        Re      = 0.1
        tau     = 1.5
        dens    = 1.0
        r_max   = torch.max(inp[b]).item()
        
        # --- Velocity de-normalization ---
        force   = force_calculation_from_R(r_max, tau=tau, Re=Re, Dens=dens)
        visc    = (tau-0.5)/3
        perm_est= (2*0.65*r_max)**2
        Kn      = 0.2
        V       = out[b, :3] * force * Kn * perm_est / visc
        
        # --- Pressure de-normalization ---
        dP      = pressure_calculation_from_R(r_max, L=out.shape[2], tau=tau, Re=Re, Dens=dens)
        P_med   = (2.0 + 3.0 * dP) / 6.0
        P_med_n = 0.15
        dP_n    = 0.2
        P       = P_med + (out[b, 3:4] - P_med_n) * dP / dP_n
    
        sample  = torch.cat([V, P], dim=0)
        all_samples.append(sample.unsqueeze(0))
        
    return torch.cat(all_samples, dim=0)


def permeability_calculation(out: torch.Tensor, 
                            inp: torch.Tensor, 
                            tau: float = 1.5, 
                            Re: float = 0.1, 
                            dens: float = 1.0) -> torch.Tensor:
    
    out_denorm  = tensor_denorm(out, inp)
    B           = out.shape[0]
    
    k_lattice   = torch.zeros(B, device=out.device, dtype=torch.float32)
    
    visc = (tau - 0.5) / 3.0
    
    for b in range(B):
        r_max        = torch.max(inp[b]).item()
        force_z      = force_calculation_from_R(R=r_max, tau=tau, Re=Re, Dens=dens)
        u_z          = out_denorm[b, 0]
        u_mean       = torch.mean(u_z)
        k_lattice[b] = 1013*(u_mean * visc) / (dens * force_z)
        
    return k_lattice