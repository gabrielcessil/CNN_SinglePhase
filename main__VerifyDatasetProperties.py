import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter

# Importe o seu leitor de dataset
from Utilities import dataset_reader as dr

def plot_dataset_property_distribution(datapath: str, 
                                       component: int = 0, 
                                       property_label: str = r"Mean Velocity $u_z$", 
                                       batch_size: int = 8, 
                                       bins: int = 10, 
                                       save_path: str = None,
                                       lim = None):
    """
    Itera sobre o dataset HDF5, calcula a média de um componente específico
    (apenas na região dos poros) para cada amostra, e plota um histograma acadêmico.
    
    Parâmetros:
        datapath: Caminho para o arquivo .h5
        component: 0 para Uz, 1 para Uy, 2 para Ux, 3 para Pressão.
        property_label: Nome do eixo X para o plot (aceita LaTeX).
        batch_size: Tamanho do batch para leitura otimizada.
        bins: Número de barras do histograma.
        save_path: Se fornecido, salva a figura neste caminho.
    """
    
    # 1. Carregar o Dataset
    print(f"Loading dataset from: {datapath}")
    dataset = dr.LazyDatasetTorch(h5_path=datapath, 
                                  list_ids=None, 
                                  x_dtype=torch.float32,
                                  y_dtype=torch.float32)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    sample_means = []
    
    # 2. Coletar os dados iterativamente (evita OOM)
    print("Extracting properties from samples...")
    with torch.no_grad():
        for batch_inp, batch_tar in loader:
            # batch_inp shape: (B, C_in, Z, Y, X)
            # batch_tar shape: (B, C_out, Z, Y, X)
            B = batch_inp.shape[0]
            
            # Máscara para ignorar a matriz sólida (considera poro onde input > 0)
            mask = batch_inp[:, 0] > 0  # Shape: (B, Z, Y, X)
            
            # Extrai apenas o componente desejado
            target_field = batch_tar[:, component] # Shape: (B, Z, Y, X)
            
            for b in range(B):
                # Extrai apenas os voxels válidos (fluidos) daquela amostra
                valid_voxels = target_field[b][mask[b]]
                
                if valid_voxels.numel() > 0:
                    sample_means.append(valid_voxels.mean().item())
                else:
                    print(f"Warning: Sample in batch has no fluid voxels.")
                    
    sample_means = np.array(sample_means)
    
    if len(sample_means) == 0:
        raise ValueError("No valid data collected. Check your dataset and mask logic.")

    # 3. Calcular Estatísticas
    mu = np.mean(sample_means)
    sigma = np.std(sample_means)
    n_samples = len(sample_means)
    
    print(f"Collected {n_samples} samples. Mean: {mu:.4e}, Std: {sigma:.4e}")

    # 4. Configuração Acadêmica do Plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    counts, bins_edges, patches = ax.hist(sample_means, bins=bins, density=True, 
                                          color='steelblue', edgecolor='black', 
                                          alpha=0.7, linewidth=1.2, label='Histogram')

   
    # 7. Linhas de Referência (Média e Mediana)
    ax.axvline(mu, color='black', linestyle='--', linewidth=1.5, label=r'Mean ($\mu$)')

    # 8. Estilização do Eixo
    ax.set_xlabel(property_label)
    ax.set_ylabel('Probability Density')
    ax.set_title('Dataset Property Distribution')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True) # Garante que o grid fique atrás das barras

    # Formatação Científica para o eixo X (útil para LBM onde u_z é da ordem de 1e-4)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter)
    
    if lim is None:
        p1, p99 = np.percentile(sample_means, [1, 99])
        margin = (p99 - p1) * 0.1
        ax.set_xlim(p1 - margin, p99 + margin)
    else:
        ax.set_xlim(lim[0], lim[1])
    
    # 9. Caixa de Estatísticas
    stats_text = '\n'.join((
        fr'$N = {n_samples}$',
        fr'$\mu = {mu:.2e}$',
        fr'$\sigma = {sigma:.2e}$'
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.95, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='right', bbox=props)

    # Legenda e Limites
    ax.legend(loc='upper right')

    # 10. Salvar ou Mostrar
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:  # Só tenta criar a pasta se o caminho tiver uma pasta definida
            os.makedirs(out_dir, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_9_random_samples_distributions(datapath: str, 
                                        component: int = 0, 
                                        property_label: str = r"Velocity $u_z$", 
                                        save_path: str = None,
                                        seed: int = 42):
    """
    Seleciona aleatoriamente 9 amostras do dataset HDF5, extrai os valores fluidos
    e plota um painel 3x3 com o histograma e a densidade (KDE) de cada amostra.
    """
    
    # 1. Carregar o Dataset (sem DataLoader, pois acessaremos os índices diretamente)
    print(f"Loading dataset from: {datapath}")
    dataset = dr.LazyDatasetTorch(h5_path=datapath, 
                                  list_ids=None, 
                                  x_dtype=torch.float32,
                                  y_dtype=torch.float32)
    
    total_samples = len(dataset)
    if total_samples < 9:
        raise ValueError(f"Dataset only has {total_samples} samples. Need at least 9.")
    
    # 2. Selecionar 9 índices aleatórios
    np.random.seed(seed)
    random_indices = np.random.choice(total_samples, size=9, replace=False)
    print(f"Selected random samples: {random_indices}")

    # 3. Configuração Acadêmica do Plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(property_label, fontsize=18, fontweight='bold')

    # 4. Iterar sobre os 9 eixos e preencher com os dados das amostras
    for i, (ax, idx) in enumerate(zip(axes.flat, random_indices)):
        # Extrai a amostra diretamente do dataset (shape: [C, Z, Y, X])
        inp, tar = dataset[idx]
        
        mask = inp[0] > 0
        
        target_field = tar[component]
        
        valid_voxels = target_field[mask].cpu().numpy()
            
        # Calcula estatísticas da amostra
        mu      = np.mean(valid_voxels)
        sigma   = np.std(valid_voxels)
        
        # Plota Histograma
        bins = min(len(valid_voxels)*0.3,1000)
        ax.hist(valid_voxels, bins=bins, density=True, 
                color='black', edgecolor='none', alpha=0.7)
        
        p1, p99 = np.percentile(valid_voxels, [1, 99])
        margin = (p99 - p1) * 0.1
        ax.set_xlim(p1 - margin, p99 + margin)
        ax.set_title(f"Sample Index: {idx}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        stats_text = '\n'.join((
            fr'$\mu = {mu:.2e}$',
            fr'$\sigma = {sigma:.2e}$'
        ))
        props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.95, 0.90, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Adiciona os rótulos X e Y apenas nas bordas para deixar o painel limpo
        if i >= 6: # Linha inferior
            ax.set_xlabel(property_label, fontsize=11)
        if i % 3 == 0: # Coluna da esquerda
            ax.set_ylabel("Density", fontsize=11)

    # 5. Salvar ou Mostrar (com a lógica de pastas já corrigida)
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir: 
            os.makedirs(out_dir, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"9-Panel Plot saved to {save_path}")
    else:
        plt.show()

# Exemplo de Execução
# ==========================================

datasets = {
    #"Cylindrical Grain":"./NN_Datasets/Train_CylinGrain_120_120_120_RotAug.h5", 
    #"Cylindrical Pores":"./NN_Datasets/Train_CylinPore_120_120_120_RotAug.h5", 
    #"Spherical Grain":"./NN_Datasets/Train_SphGrain_120_120_120_RotAug.h5", 
    #"Spherical Pores":"./NN_Datasets/Train_SphPore_120_120_120_RotAug.h5", 
    
    
    
    #"Spherical Pores (KC-Maximas)": "./NN_Datasets/Train_SphPore_120_120_120_RotAug_KlocRMAX.h5",
    #"Spherical Grains (KC-Maximas)": "./NN_Datasets/Train_SphGrain_120_120_120_RotAug_KlocRMAX.h5",
    #"Bentheimer (KC-Maximas)": "./NN_Datasets/Test_Oliveira_Bentheimer_120_120_120_RotAug_KlocRMAX.h5",
    
    #"Parker (Rmaximas^2)": "./NN_Datasets/Test_Oliveira_Parker_120_120_120_RotAug_Rmaximas2.h5",
    #"Parker (KC-Maximas)": "./NN_Datasets/Test_Oliveira_Parker_120_120_120_RotAug_KC.h5"
    "Parker (Rmax^2)": "./NN_Datasets/Test_Oliveira_Parker_120_120_120_RotAug_Rmax.h5"
    
    #"Spherical Pores (RT-Maximas)": "./NN_Datasets/Train_SphPore_120_120_120_RotAug_KlocRMAXTHUMB.h5",
    #"Spherical Grains (RT-Maximas)": "./NN_Datasets/Train_SphGrain_120_120_120_RotAug_KlocRMAXTHUMB.h5",
    #"Parker (RT-Maximas)": "./NN_Datasets/Test_Oliveira_Parker_120_120_120_RotAug_KlocRMAXTHUMB.h5",
    #"Bentheimer (RT-Maximas)": "./NN_Datasets/Test_Bentheimer_Parker_120_120_120_RotAug_KlocRMAXTHUMB.h5",
    
}


for dataset_name, datapath in datasets.items():
    
    # Exemplo 1: Distribuição da Velocidade Uz (componente 0)
    plot_dataset_property_distribution(datapath=datapath, 
                                       component=0, 
                                       property_label=r"Mean Velocity $u_z$ [Pre-Processed]",
                                       batch_size=4,
                                       save_path=dataset_name+"_Distribution_Uz_Dataset.png",
                                       lim=(0,0.42))
    
    plot_9_random_samples_distributions(datapath=datapath, 
                                       component=0, 
                                       property_label=r"Mean Velocity $u_z$ [Pre-Processed]",
                                       save_path=dataset_name+"_Distribution_Uz_Samples.png")
    
# Exemplo 2: Distribuição da Pressão (componente 3)
# plot_dataset_property_distribution(datapath=datapath, 
#                                    component=3, 
#                                    property_label=r"Mean Pressure $P$",
#                                    batch_size=4,
#                                    save_path="Distribution_Pressure.png")