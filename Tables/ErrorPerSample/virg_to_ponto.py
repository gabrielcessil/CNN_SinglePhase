import pandas as pd

arquivo_entrada = 'ErrorMetrics_Comp5_Ko_et_al_D3.csv'
arquivo_saida = 'ErrorMetrics_Comp5_Ko_et_al_D3_PONTO.csv'

# Lê o CSV: vírgula separa as colunas e os decimais usam vírgula.
df = pd.read_csv(
    arquivo_entrada,
    sep=',',
    decimal=',',
    encoding='utf-8-sig'
)

# Converte as colunas numéricas para tipos numéricos.
# As colunas textuais, como Dataset, permanecem inalteradas.
for coluna in df.columns:
    if coluna not in ['Dataset']:
        df[coluna] = pd.to_numeric(df[coluna], errors='raise')

# Salva com vírgula como separador e ponto como separador decimal.
df.to_csv(
    arquivo_saida,
    sep=',',
    decimal='.',
    index=False,
    encoding='utf-8'
)

print(f'Conversão concluída: {arquivo_saida}')
print('\nTipos de dados após a conversão:')
print(df.dtypes)
