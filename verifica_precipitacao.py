#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verifica_precipitacao.py
========================

Verificacao de previsoes de precipitacao acumulada em 24 h contra um campo
observado (analise) de precipitacao acumulada em 24 h definido em uma malha.

Fluxo:
  1. Le previsao e observacao (NetCDF, GRIB/GRIB2, GeoTIFF ou CSV de pontos).
  2. Coloca previsao e observacao na MESMA malha (regrid da previsao para a
     grade da observacao, por interpolacao bilinear ou "nearest").
  3. Aplica mascara de dados validos (remove NaN e regiao fora do dominio).
  4. Calcula metricas de verificacao:
        - Continuas .......... vies (BIAS), MAE, RMSE, correlacao de Pearson,
                               n de pares.
        - Categoricas ........ tabela de contingencia por limiar (ex.: 1, 10,
                               25, 50 mm) -> POD, FAR, POFD, CSI, ETS, HSS,
                               frequency bias (FBIAS), acuracia.
        - Espaciais .......... FSS (Fractions Skill Score) por limiar e por
                               escala de vizinhanca.
  5. Salva um CSV com todas as metricas e (opcional) figuras/mapas.

Uso basico
----------
    python verifica_precipitacao.py \
        --previsao  prev_24h.nc \
        --observacao obs_24h.nc \
        --var-prev  tp --var-obs  precip \
        --limiares  1 10 25 50 \
        --escalas-fss 1 3 5 11 \
        --saida     resultados/

Dependencias
------------
    numpy, pandas, xarray            (obrigatorias)
    scipy                            (interpolacao de regrid)
    netCDF4                          (NetCDF)
    cfgrib / eccodes                 (GRIB) .... opcional
    rasterio                         (GeoTIFF) . opcional
    matplotlib                       (figuras) .. opcional

Instalacao rapida:
    pip install numpy pandas xarray scipy netCDF4 matplotlib
    pip install cfgrib rasterio            # se for usar GRIB / GeoTIFF

Autor: gerado para o projeto Verif_prec.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. LEITURA DE DADOS
# ---------------------------------------------------------------------------
@dataclass
class Campo:
    """Campo 2D de precipitacao georreferenciado (grade regular lat/lon)."""
    dados: np.ndarray          # shape (nlat, nlon), mm em 24 h
    lats: np.ndarray           # shape (nlat,)  crescente ou decrescente
    lons: np.ndarray           # shape (nlon,)
    nome: str = ""

    def __post_init__(self):
        self.dados = np.asarray(self.dados, dtype=float)
        self.lats = np.asarray(self.lats, dtype=float)
        self.lons = np.asarray(self.lons, dtype=float)
        if self.dados.shape != (self.lats.size, self.lons.size):
            raise ValueError(
                f"[{self.nome}] shape {self.dados.shape} incompativel com "
                f"lat({self.lats.size}) x lon({self.lons.size})"
            )


def _norm_lons(lons: np.ndarray) -> np.ndarray:
    """Converte longitudes 0..360 para -180..180 para casar dominios."""
    lons = np.asarray(lons, dtype=float)
    return np.where(lons > 180.0, lons - 360.0, lons)


def _detecta_coord(ds, candidatos):
    for c in candidatos:
        if c in ds.coords or c in ds.variables:
            return c
    return None


def le_netcdf(caminho: str, var: Optional[str] = None,
              tempo: Optional[int] = None, nome: str = "") -> Campo:
    import xarray as xr
    ds = xr.open_dataset(caminho)

    if var is None:
        # escolhe a primeira variavel com >= 2 dims
        cand = [v for v in ds.data_vars if ds[v].ndim >= 2]
        if not cand:
            raise ValueError(f"Nenhuma variavel 2D encontrada em {caminho}")
        var = cand[0]
        print(f"  [aviso] variavel nao informada; usando '{var}'")
    da = ds[var]

    latname = _detecta_coord(ds, ["lat", "latitude", "y", "Y", "XLAT"])
    lonname = _detecta_coord(ds, ["lon", "longitude", "x", "X", "XLONG"])
    if latname is None or lonname is None:
        raise ValueError(f"Nao encontrei coords lat/lon em {caminho}")

    # reduz dimensoes de tempo/nivel se existirem
    for d in list(da.dims):
        if d in (latname, lonname):
            continue
        if tempo is not None and d.lower() in ("time", "t", "valid_time"):
            da = da.isel({d: tempo})
        else:
            da = da.isel({d: 0})

    da = da.transpose(latname, lonname)
    return Campo(da.values, ds[latname].values, _norm_lons(ds[lonname].values),
                 nome=nome or var)


def le_grib(caminho: str, var: Optional[str] = None, nome: str = "") -> Campo:
    import xarray as xr
    ds = xr.open_dataset(caminho, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    return _campo_de_xr(ds, var, nome, caminho)


def _campo_de_xr(ds, var, nome, caminho):
    if var is None:
        cand = [v for v in ds.data_vars if ds[v].ndim >= 2]
        if not cand:
            raise ValueError(f"Nenhuma variavel 2D em {caminho}")
        var = cand[0]
        print(f"  [aviso] variavel nao informada; usando '{var}'")
    da = ds[var]
    latname = _detecta_coord(ds, ["lat", "latitude", "y"])
    lonname = _detecta_coord(ds, ["lon", "longitude", "x"])
    for d in list(da.dims):
        if d not in (latname, lonname):
            da = da.isel({d: 0})
    da = da.transpose(latname, lonname)
    return Campo(da.values, ds[latname].values, _norm_lons(ds[lonname].values),
                 nome=nome or var)


def le_geotiff(caminho: str, nome: str = "") -> Campo:
    import rasterio
    with rasterio.open(caminho) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        nrows, ncols = arr.shape
        t = src.transform
        # centros das celulas
        cols = np.arange(ncols)
        rows = np.arange(nrows)
        lons = t.c + t.a * (cols + 0.5)
        lats = t.f + t.e * (rows + 0.5)
    # rasterio retorna linhas de cima para baixo -> lat decrescente
    return Campo(arr, lats, _norm_lons(lons), nome=nome or "geotiff")


def le_csv(caminho: str, nome: str = "", grade_ref: Optional[Campo] = None) -> Campo:
    """CSV com colunas lat, lon, valor. Se grade_ref for dada, os pontos sao
    interpolados para essa grade; senao monta uma grade regular a partir dos
    valores unicos de lat/lon."""
    df = pd.read_csv(caminho)
    cols = {c.lower(): c for c in df.columns}
    clat = cols.get("lat") or cols.get("latitude")
    clon = cols.get("lon") or cols.get("longitude")
    cval = (cols.get("valor") or cols.get("value") or cols.get("precip")
            or cols.get("prec") or cols.get("pr"))
    if not (clat and clon and cval):
        raise ValueError(f"CSV {caminho} precisa de colunas lat, lon e valor")
    lat = df[clat].to_numpy(float)
    lon = _norm_lons(df[clon].to_numpy(float))
    val = df[cval].to_numpy(float)

    if grade_ref is not None:
        from scipy.interpolate import griddata
        LON, LAT = np.meshgrid(grade_ref.lons, grade_ref.lats)
        z = griddata((lon, lat), val, (LON, LAT), method="linear")
        return Campo(z, grade_ref.lats, grade_ref.lons, nome=nome or "csv")

    lats = np.unique(lat)
    lons = np.unique(lon)
    grid = np.full((lats.size, lons.size), np.nan)
    ilat = {v: i for i, v in enumerate(lats)}
    ilon = {v: i for i, v in enumerate(lons)}
    for la, lo, v in zip(lat, lon, val):
        grid[ilat[la], ilon[lo]] = v
    return Campo(grid, lats, lons, nome=nome or "csv")


def carrega(caminho: str, var: Optional[str] = None,
            tempo: Optional[int] = None, nome: str = "",
            grade_ref: Optional[Campo] = None) -> Campo:
    ext = os.path.splitext(caminho)[1].lower()
    if ext in (".nc", ".nc4", ".netcdf", ".cdf"):
        return le_netcdf(caminho, var, tempo, nome)
    if ext in (".grib", ".grib2", ".grb", ".grb2", ".gb2"):
        return le_grib(caminho, var, nome)
    if ext in (".tif", ".tiff", ".geotiff"):
        return le_geotiff(caminho, nome)
    if ext in (".csv", ".txt", ".dat"):
        return le_csv(caminho, nome, grade_ref)
    raise ValueError(f"Extensao nao suportada: {ext}")


# ---------------------------------------------------------------------------
# 2. REGRID (coloca previsao na grade da observacao)
# ---------------------------------------------------------------------------
def regrid(origem: Campo, destino: Campo, metodo: str = "linear") -> Campo:
    """Interpola o campo 'origem' para a grade de 'destino'.
    metodo: 'linear' (bilinear) ou 'nearest'."""
    from scipy.interpolate import RegularGridInterpolator

    # RegularGridInterpolator exige eixos estritamente crescentes
    lats_o, dados = origem.lats, origem.dados
    if lats_o[0] > lats_o[-1]:
        lats_o = lats_o[::-1]
        dados = dados[::-1, :]
    lons_o = origem.lons
    if lons_o[0] > lons_o[-1]:
        lons_o = lons_o[::-1]
        dados = dados[:, ::-1]

    interp = RegularGridInterpolator(
        (lats_o, lons_o), dados,
        method=metodo, bounds_error=False, fill_value=np.nan)

    LON, LAT = np.meshgrid(destino.lons, destino.lats)
    pts = np.column_stack([LAT.ravel(), LON.ravel()])
    z = interp(pts).reshape(LAT.shape)
    return Campo(z, destino.lats, destino.lons, nome=origem.nome + "_regrid")


# ---------------------------------------------------------------------------
# 3. METRICAS
# ---------------------------------------------------------------------------
def metricas_continuas(prev: np.ndarray, obs: np.ndarray) -> dict:
    """Metricas continuas sobre pares validos (sem NaN)."""
    m = np.isfinite(prev) & np.isfinite(obs)
    p, o = prev[m], obs[m]
    n = p.size
    if n == 0:
        return {k: np.nan for k in
                ("n", "media_prev", "media_obs", "bias", "mae", "rmse",
                 "corr", "bias_multiplicativo")}
    dif = p - o
    corr = np.corrcoef(p, o)[0, 1] if n > 1 and p.std() > 0 and o.std() > 0 else np.nan
    return {
        "n": int(n),
        "media_prev": float(p.mean()),
        "media_obs": float(o.mean()),
        "bias": float(dif.mean()),                       # vies medio (mm)
        "mae": float(np.abs(dif).mean()),                # erro absoluto medio
        "rmse": float(np.sqrt((dif ** 2).mean())),       # raiz do erro quad. medio
        "corr": float(corr),                             # Pearson
        "bias_multiplicativo": float(p.sum() / o.sum()) if o.sum() > 0 else np.nan,
    }


def tabela_contingencia(prev: np.ndarray, obs: np.ndarray, limiar: float):
    """Retorna (hits, misses, false_alarms, correct_negatives) para prec >= limiar."""
    m = np.isfinite(prev) & np.isfinite(obs)
    p = prev[m] >= limiar
    o = obs[m] >= limiar
    hits = int(np.sum(p & o))
    misses = int(np.sum(~p & o))
    fa = int(np.sum(p & ~o))
    cn = int(np.sum(~p & ~o))
    return hits, misses, fa, cn


def scores_categoricos(prev: np.ndarray, obs: np.ndarray, limiar: float) -> dict:
    a, c, b, d = tabela_contingencia(prev, obs, limiar)  # a=hits b=FA c=miss d=CN
    n = a + b + c + d
    def sd(x, y):  # divisao segura
        return x / y if y else np.nan
    pod = sd(a, a + c)                       # prob. de deteccao (0..1, 1 otimo)
    far = sd(b, a + b)                        # razao de alarme falso (0..1, 0 otimo)
    pofd = sd(b, b + d)                       # prob. de falso alarme
    csi = sd(a, a + b + c)                    # indice critico de sucesso (TS)
    fbias = sd(a + b, a + c)                  # vies de frequencia (1 otimo)
    acc = sd(a + d, n)                        # acuracia
    # ETS (Gilbert Skill Score)
    a_ref = sd((a + b) * (a + c), n)
    ets = sd(a - a_ref, a + b + c - a_ref)   # -1/3..1, 0 = sem skill
    # Heidke Skill Score
    esp = sd((a + b) * (a + c) + (c + d) * (b + d), n)
    hss = sd(a + d - esp, n - esp)
    return {
        "limiar_mm": limiar,
        "hits": a, "false_alarms": b, "misses": c, "correct_neg": d, "n": n,
        "POD": pod, "FAR": far, "POFD": pofd, "CSI": csi,
        "FBIAS": fbias, "ETS": ets, "HSS": hss, "acuracia": acc,
    }


def _integral_imagem(campo_bin: np.ndarray) -> np.ndarray:
    """Soma acumulada 2D (summed-area table) com borda de zeros."""
    return np.pad(np.cumsum(np.cumsum(campo_bin, axis=0), axis=1),
                  ((1, 0), (1, 0)), mode="constant")


def _fracoes(campo_bin: np.ndarray, n: int) -> np.ndarray:
    """Fracao de pixels 'excedentes' em janela quadrada (2n+1) via imagem integral."""
    S = _integral_imagem(campo_bin)
    H, W = campo_bin.shape
    out = np.empty_like(campo_bin, dtype=float)
    for i in range(H):
        i0, i1 = max(0, i - n), min(H, i + n + 1)
        for j in range(W):
            j0, j1 = max(0, j - n), min(W, j + n + 1)
            soma = S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]
            area = (i1 - i0) * (j1 - j0)
            out[i, j] = soma / area
    return out


def fss(prev: np.ndarray, obs: np.ndarray, limiar: float, escala: int) -> float:
    """Fractions Skill Score. escala = tamanho da vizinhanca em pixels (impar).
    Retorna 1 = perfeito, 0 = sem habilidade."""
    m = np.isfinite(prev) & np.isfinite(obs)
    p = (np.where(m, prev, 0.0) >= limiar) & m
    o = (np.where(m, obs, 0.0) >= limiar) & m
    n = escala // 2
    fp = _fracoes(p.astype(float), n)
    fo = _fracoes(o.astype(float), n)
    fp = fp[m]
    fo = fo[m]
    mse = np.mean((fp - fo) ** 2)
    mse_ref = np.mean(fp ** 2) + np.mean(fo ** 2)
    if mse_ref == 0:
        return np.nan
    return float(1.0 - mse / mse_ref)


# ---------------------------------------------------------------------------
# 4. RELATORIO / FIGURAS
# ---------------------------------------------------------------------------
def salva_figuras(prev: Campo, obs: Campo, saida: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [aviso] matplotlib indisponivel, pulando figuras ({e})")
        return
    dif = prev.dados - obs.dados
    ext = [obs.lons.min(), obs.lons.max(), obs.lats.min(), obs.lats.max()]
    validos = np.concatenate([obs.dados[np.isfinite(obs.dados)],
                              prev.dados[np.isfinite(prev.dados)]])
    vmax = float(np.nanpercentile(validos, 99)) if validos.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for a, campo, tit, cmap, vlim in [
        (ax[0], obs.dados, "Observado (24h)", "Blues", (0, vmax)),
        (ax[1], prev.dados, "Previsto (24h)", "Blues", (0, vmax)),
        (ax[2], dif, "Previsto - Observado", "RdBu_r", (-vmax, vmax)),
    ]:
        im = a.imshow(campo, origin="lower", extent=ext, aspect="auto",
                      cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        a.set_title(tit); a.set_xlabel("lon"); a.set_ylabel("lat")
        fig.colorbar(im, ax=a, shrink=0.85, label="mm")
    fig.tight_layout()
    fig_path = os.path.join(saida, "mapas_verificacao.png")
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"  figura salva: {fig_path}")


# ---------------------------------------------------------------------------
# 5. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------
def executa(args) -> int:
    os.makedirs(args.saida, exist_ok=True)

    print("Lendo observacao ...")
    obs = carrega(args.observacao, args.var_obs, args.tempo_obs, nome="obs")

    print("Lendo previsao ...")
    # CSV de previsao pode ser interpolado direto para a grade da obs
    prev = carrega(args.previsao, args.var_prev, args.tempo_prev,
                   nome="prev", grade_ref=obs)

    # Regrid da previsao para a grade da observacao (se necessario)
    mesma_grade = (prev.lats.shape == obs.lats.shape and
                   prev.lons.shape == obs.lons.shape and
                   np.allclose(prev.lats, obs.lats) and
                   np.allclose(prev.lons, obs.lons))
    if not mesma_grade:
        print(f"Regrid da previsao para a grade da observacao "
              f"({obs.lats.size}x{obs.lons.size}) via '{args.metodo_regrid}' ...")
        prev = regrid(prev, obs, metodo=args.metodo_regrid)
    else:
        print("Previsao e observacao ja estao na mesma grade.")

    P, O = prev.dados, obs.dados

    # ---- metricas continuas
    cont = metricas_continuas(P, O)
    print("\n=== METRICAS CONTINUAS ===")
    for k, v in cont.items():
        print(f"  {k:>20s}: {v}")
    pd.DataFrame([cont]).to_csv(
        os.path.join(args.saida, "metricas_continuas.csv"), index=False)

    # ---- metricas categoricas
    linhas_cat = [scores_categoricos(P, O, t) for t in args.limiares]
    df_cat = pd.DataFrame(linhas_cat)
    print("\n=== METRICAS CATEGORICAS (por limiar) ===")
    print(df_cat.to_string(index=False))
    df_cat.to_csv(os.path.join(args.saida, "metricas_categoricas.csv"),
                  index=False)

    # ---- FSS
    if args.escalas_fss:
        linhas_fss = []
        for t in args.limiares:
            for s in args.escalas_fss:
                linhas_fss.append(
                    {"limiar_mm": t, "escala_px": s, "FSS": fss(P, O, t, s)})
        df_fss = pd.DataFrame(linhas_fss)
        print("\n=== FSS (Fractions Skill Score) ===")
        print(df_fss.pivot(index="limiar_mm", columns="escala_px",
                           values="FSS").to_string())
        df_fss.to_csv(os.path.join(args.saida, "metricas_fss.csv"), index=False)

    # ---- figuras
    if not args.sem_figuras:
        salva_figuras(prev, obs, args.saida)

    print(f"\nResultados salvos em: {os.path.abspath(args.saida)}")
    return 0


def constroi_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verificacao de previsao de precipitacao acumulada 24h "
                    "contra observacao em malha.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--previsao", required=True, help="arquivo da previsao")
    p.add_argument("--observacao", required=True, help="arquivo da observacao")
    p.add_argument("--var-prev", default=None,
                   help="nome da variavel de precip na previsao")
    p.add_argument("--var-obs", default=None,
                   help="nome da variavel de precip na observacao")
    p.add_argument("--tempo-prev", type=int, default=None,
                   help="indice de tempo a extrair da previsao (se houver)")
    p.add_argument("--tempo-obs", type=int, default=None,
                   help="indice de tempo a extrair da observacao (se houver)")
    p.add_argument("--limiares", type=float, nargs="+",
                   default=[1, 10, 25, 50],
                   help="limiares (mm/24h) para as metricas categoricas e FSS")
    p.add_argument("--escalas-fss", type=int, nargs="*",
                   default=[1, 3, 5, 11],
                   help="tamanhos de vizinhanca (px, impar) para o FSS; "
                        "vazio desativa o FSS")
    p.add_argument("--metodo-regrid", choices=["linear", "nearest"],
                   default="linear", help="metodo de interpolacao no regrid")
    p.add_argument("--saida", default="resultados_verificacao",
                   help="pasta de saida")
    p.add_argument("--sem-figuras", action="store_true",
                   help="nao gerar mapas PNG")
    return p


def main(argv=None) -> int:
    args = constroi_parser().parse_args(argv)
    try:
        return executa(args)
    except Exception as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
