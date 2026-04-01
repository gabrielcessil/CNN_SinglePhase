import torch
from   torch.utils.data import DataLoader
import torch.nn as nn
from Utilities import dataset_reader as dr
from Utilities import loss_functions as lf
from Danny_Original.architecture import Danny_KerasModel

datapath   = "../NN_Datasets/PressureDriven/Train_Danny_120_120_120_Pressure.h5" 
batch_size = 4


baseline_model = Danny_KerasModel()


dataset = dr.LazyDatasetTorch(
    h5_path=datapath, 
    list_ids=None, 
    x_dtype=torch.float32,
    y_dtype=torch.float32
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


loss_fn = lf.MassConservation()


def create_poiseuville_test_tensor(size=120, radius=40):
    # Criar grid de coordenadas (Z, Y, X)
    z = torch.linspace(0, 1, size)
    y = torch.linspace(-size//2, size//2, size)
    x = torch.linspace(-size//2, size//2, size)
    
    # meshgrid para 3D
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    
    # Calcular raio a partir do centro do duto (eixo Z)
    r_sq = grid_x**2 + grid_y**2
    R_sq = radius**2
    
    # 1. Velocidade Z (Parabólica)
    vz = 1.0 * (1 - r_sq / R_sq)
    vz[r_sq > R_sq] = 0  # Condição de não-deslizamento (fora do cilindro)
    
    # 2. Velocidades X e Y (Zero em Poiseuville)
    vy = torch.zeros_like(vz)
    vx = torch.zeros_like(vz)
    
    # 3. Densidade/Pressão (Linear em Z)
    # rho(z) = rho_in - (rho_in - rho_out) * z
    rho = 1.1 - 0.2 * grid_z 
    rho[r_sq > R_sq] = 1.0 # Densidade no sólido (opcional)
    
    # Unir em um tensor (B, C, Z, Y, X) -> B=1, C=4
    test_tensor = torch.stack([vz, vy, vx, rho/3.0], dim=0).unsqueeze(0)
    
    return test_tensor

input_test  = create_poiseuville_test_tensor(size=256, radius=100)
loss        = loss_fn(input_test,input_test)
print(f"Mass Conservation Loss: {loss.item():.10e}")


#def tensor_denorm(tensor):
    
    

with torch.no_grad():
    for batch_idx, (inp, tar) in enumerate(loader):
        
        inp = inp.to(dtype=torch.float32)
        tar = tar.to(dtype=torch.float32)
        #out = model.predict(inp)
        
        loss = loss_fn(tar,tar)
        print(f"Mass Conservation Loss (Target): {loss.item():.10e}")

