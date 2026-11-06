import numpy as np
def comparar_arquivos_raw_4_componentes(dir1, dir2, filename, grid_shape=(120, 120, 120), dtype=np.float64):
    """
    Lê dois arquivos .raw de 4 componentes (ex: vetor + escalar) e os compara.
    Assume formato intercalado (shape final: Z, Y, X, 4).
    """
    path1 = f"{dir1.rstrip('/')}/{filename}"
    path2 = f"{dir2.rstrip('/')}/{filename}"
    
    print(f"Lendo: {path1}")
    print(f"Lendo: {path2}")
    
    # 1. Leitura do arquivo bruto completo
    flat1 = np.fromfile(path1, dtype=dtype)
    flat2 = np.fromfile(path2, dtype=dtype)
    
    # O tamanho esperado total é 120 * 120 * 120 * 4
    expected_size = grid_shape[0] * grid_shape[1] * grid_shape[2] * 4
    if flat1.size != expected_size or flat2.size != expected_size:
        print(f"Aviso: Tamanho do arquivo incompatível! Esperado {expected_size} floats, lido {flat1.size}.")
    
    # 2. Reshape para a malha 3D + a dimensão dos 4 componentes
    arr1 = flat1.reshape((*grid_shape, 4))
    arr2 = flat2.reshape((*grid_shape, 4))
    
    # 3. Comparação global
    sao_exatamente_iguais = np.array_equal(arr1, arr2)
    print("\n--- Resultados Globais ---")
    print(f"Igualdade exata (todos os 4 componentes): {sao_exatamente_iguais}")
    
    # 4. Análise componente a componente
    component_names = ["Componente 0", "Componente 1", "Componente 2", "Componente 3"]
    # Se forem velocidade e pressão, por exemplo, você pode renomear para:
    # component_names = ["U", "V", "W", "Pressão"]
    
    diff = np.abs(arr1 - arr2)
    
    for i in range(4):
        c_diff = diff[..., i]
        max_diff = np.max(c_diff)
        mean_diff = np.mean(c_diff)
        equal_c = np.array_equal(arr1[..., i], arr2[..., i])
        
        print(f"\n[{component_names[i]}]")
        print(f"  - Exatamente igual: {equal_c}")
        print(f"  - Maior diferença absoluta: {max_diff:.4e}")
        print(f"  - Diferença média absoluta: {mean_diff:.4e}")
        

    # 5. Verificação global com tolerância
    proximos = np.allclose(arr1, arr2, rtol=1e-07, atol=1e-12)
    print(f"\nEquivalentes dentro da tolerância global (allclose): {proximos}")

    return arr1, arr2, diff

# Exemplo de uso:
dir_A = "./Desktop_cpu"
dir_B = "./Cluster_cpu"
campo = "Start.00000.raw"
arr1, arr2, diff = comparar_arquivos_raw_4_componentes(dir_A, dir_B, campo)