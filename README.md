# Verif-prec — Verificação de precipitação (Eta / CPTEC vs MERGE)

Scripts para verificação de previsões de precipitação **acumulada em 24 h**
contra um campo observado definido em malha.

## Scripts

### `verifica_eta.py`
Verificação das rodadas do modelo **Eta** (jaci e XC50/operacional) contra a
base observada **MERGE**, na resolução do observado (**10 km**).

- Lê `nc/` (jaci, precip em **metros** → ×1000 = mm) usando o `PREC-ACUM24h`;
- Lê `nc_oper/` (XC50, **mm**) e **acumula 24 h** a partir dos passos sub-diários;
- Usa `nc_merge/` (**mm**, 10 km) como observação;
- Regrida a previsão (8 km) para a grade do MERGE (10 km);
- Janelas de 24 h **12Z–12Z** e **00Z–00Z**;
- Métricas contínuas (viés, MAE, RMSE, correlação), categóricas por limiar
  (POD, FAR, CSI, ETS, HSS, FBIAS) e **FSS**;
- Saídas: `placar_por_modelo.csv`, `scores_por_prazo.csv`,
  `scores_por_rodada.csv` e `mapas_<modelo>.png`.

```bash
# inspecionar a estrutura interna de um arquivo:
python verifica_eta.py --inspecionar caminho/arquivo.nc

# verificação completa:
python verifica_eta.py --base /caminho/Verif-prec --saida resultados_eta --janelas 12 0
```

Estrutura de dados esperada:

```
Verif-prec/
  nc/YYYYMMDDHH/PREC-ACUM24h_YYYYMMDD.nc   # jaci (metros)
  nc_oper/YYYYMMDDHH/PREC_YYYYMMDD.nc      # XC50 (mm, sub-diario)
  nc_merge/MERGE_*.nc                       # observado (mm, 10 km)
```

### `verifica_precipitacao.py`
Verificação genérica de dois campos de precipitação 24 h (NetCDF, GRIB,
GeoTIFF ou CSV), com regrid para grade comum e as mesmas famílias de métricas.

## Dependências

```bash
pip install numpy pandas xarray scipy netCDF4 matplotlib
# opcionais: cfgrib (GRIB), rasterio (GeoTIFF)
```
