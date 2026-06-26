 
import torch
import numpy as np
from torch.utils.data import DataLoader
from Utilities import dataset_reader as dr

def analyze_porosity_slices(datasets_dict: dict, batch_size: int = 4):
    
    results = {}

    for dataset_name, datapath in datasets_dict.items():
        print(f"--- Analisando: {dataset_name} ---")
        dataset = dr.LazyDatasetTorch(h5_path=datapath, list_ids=None, 
                                      x_dtype=torch.float32, y_dtype=torch.float32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        ratios = []
        
        with torch.no_grad():
            for batch_inp, _ in loader:
                inp =  batch_inp[:, 0].cpu().numpy()
                mask = (inp > 0)
                
                
                for b in range(mask.shape[0]):
                    sample = mask[b]
                    
                    slice_porosities  = np.mean(sample, axis=(1, 2))
                    
                    max_porosity      = np.max(slice_porosities)
                    
                    min_porosity      = np.min(slice_porosities)
                                        
                    if min_porosity > 0:
                        ratio = min_porosity / max_porosity
                        ratios.append(ratio)
                    
        results[dataset_name] = np.array(ratios)
        print(f"Processado: {len(ratios)} amostras.")
        
    return results

# ==========================================
# Exemplo de Execução
# ==========================================
datasets = {
    "Spherical Pores":  "../NN_Datasets_Grad/Valid_SphPore_SAug_DNorm.h5",
    "Spherical Grains": "../NN_Datasets_Grad/Valid_SphGrain_SAug_DNorm.h5",
}

analysis_data = analyze_porosity_slices(datasets)


for dataset, ratios in analysis_data.items():
    
    print(dataset)
    print(ratios)
    print()


datasets = {
    "Leopard":           "../NN_Datasets_Grad/Valid_Oliveira_Leopard_SAug_DNorm.h5",
    "Castle Gate":       "../NN_Datasets_Grad/Valid_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":  "../NN_Datasets_Grad/Valid_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray": "../NN_Datasets_Grad/Valid_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "Berea Buff":        "../NN_Datasets_Grad/Valid_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Berea":             "../NN_Datasets_Grad/Valid_Oliveira_Berea_SAug_DNorm.h5",
    "Bentheimer":        "../NN_Datasets_Grad/Valid_Oliveira_Bentheimer_SAug_DNorm.h5",
}

analysis_data = analyze_porosity_slices(datasets)


for dataset, ratios in analysis_data.items():
    
    print(dataset)
    print(ratios)
    print()