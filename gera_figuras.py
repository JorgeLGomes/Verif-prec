#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gera_figuras.py
===============

Gera TODAS as figuras da verificacao a partir do binario 'verificacao.pkl'
salvo pelo verifica_eta.py - assim nao e' preciso reler os NetCDF nem refazer
a verificacao so para replotar/ajustar figuras.

Produz, para cada janela (12Z, 00Z):
    - continuas_<janela>.png ..... vies, MAE, RMSE, correlacao, vies mult., N
    - categoricas_<janela>.png ... POD, FAR, CSI, ETS, HSS, FBIAS (cor=limiar)
    - fss_<janela>.png ........... FSS por escala de vizinhanca (cor=limiar)
e, por modelo:
    - mapas_<modelo>.png ......... obs media, prev media e vies medio (mm)

Convencao: jaci -> linha CONTINUA ('-');  XC50 -> linha TRACEJADA ('--').

Uso:
    python gera_figuras.py --binario resultados_eta/verificacao.pkl
    python gera_figuras.py --binario resultados_eta/verificacao.pkl --figuras figs/
    python gera_figuras.py --binario resultados_eta/verificacao.pkl --janelas 12Z

Depende de: numpy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ESTILO = {"jaci": "-", "xc50": "--"}
MARCA = {"jaci": "o", "xc50": "s"}


def _estilo(m):
    return ESTILO.get(m, "-"), MARCA.get(m, "o")


def _cores_limiar(limiares):
    """Cores de alto contraste por limiar, preservando a ordem.
    Usa 'turbo' (azul->ciano->verde->amarelo->vermelho): ordenada e bem
    mais contrastante que viridis. Para poucos limiares usa cores nitidas."""
    n = len(limiares)
    if n <= 8:
        base = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                "#17becf", "#8c564b", "#000000"]  # qualitativa, forte contraste
        return {t: base[i % len(base)] for i, t in enumerate(limiares)}
    import numpy as _np
    cores = plt.cm.turbo(_np.linspace(0.02, 0.98, n))
    return {t: c for t, c in zip(limiares, cores)}


# ------------------------------------------------------------------ linhas ---
def plota_continuas(df, janela, saida):
    metricas = [("cont_bias", "Vies (mm)"), ("cont_mae", "MAE (mm)"),
                ("cont_rmse", "RMSE (mm)"), ("cont_corr", "Correlacao"),
                ("cont_vies_mult", "Vies multiplicativo"),
                ("cont_n", "N de pares")]
    d = df.drop_duplicates(subset=["modelo", "lead"]).copy()
    modelos = sorted(d.modelo.unique())
    _cmod = {"jaci": "#1f77b4", "xc50": "#d62728"}
    paleta = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    cor = {m: _cmod.get(m, paleta[i % len(paleta)]) for i, m in enumerate(modelos)}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, rot) in zip(axs.ravel(), metricas):
        if col not in d.columns:
            ax.axis("off"); continue
        for m in modelos:
            sub = d[d.modelo == m].sort_values("lead")
            ls, mk = _estilo(m)
            ax.plot(sub.lead, sub[col], ls, marker=mk, color=cor[m],
                    markersize=5, label=m)
        ax.set_title(rot); ax.set_xlabel("Prazo (D+)"); ax.grid(alpha=0.3)
        ax.set_xticks(sorted(d.lead.unique()))
        if col == "cont_bias":
            ax.axhline(0, color="grey", lw=0.8, ls=":")
        if col == "cont_vies_mult":
            ax.axhline(1, color="grey", lw=0.8, ls=":")
    h = [Line2D([0], [0], color="black", linestyle=_estilo(m)[0],
                marker=_estilo(m)[1],
                label=f"{m} ({'continua' if _estilo(m)[0]=='-' else 'tracejada'})")
         for m in modelos]
    fig.tight_layout(rect=[0, 0, 1, 0.90])   # reserva faixa no topo
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.985),
               fontsize=9, ncol=len(h), framealpha=0.9, borderaxespad=0.3)
    fig.suptitle(f"Metricas continuas x prazo - janela {janela}", fontsize=13,
                 x=0.5, y=0.995)
    cam = os.path.join(saida, f"continuas_{janela}.png")
    fig.savefig(cam, dpi=130); plt.close(fig); print(f"  {cam}")


def plota_categoricas(df, janela, saida):
    metricas = [("POD", "POD"), ("FAR", "FAR"), ("CSI", "CSI"),
                ("ETS", "ETS"), ("HSS", "HSS"), ("FBIAS", "Vies de freq.")]
    modelos = sorted(df.modelo.unique())
    limiares = sorted(df.limiar_mm.unique())
    cor = _cores_limiar(limiares)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, rot) in zip(axs.ravel(), metricas):
        for m in modelos:
            ls, mk = _estilo(m)
            for t in limiares:
                sub = df[(df.modelo == m) & (df.limiar_mm == t)].sort_values("lead")
                if sub.empty:
                    continue
                ax.plot(sub.lead, sub[col], ls, marker=mk, color=cor[t],
                        markersize=4, lw=1.8)
        ax.set_title(rot); ax.set_xlabel("Prazo (D+)"); ax.grid(alpha=0.3)
        ax.set_xticks(sorted(df.lead.unique()))
        if col == "FBIAS":
            ax.axhline(1, color="grey", lw=0.8, ls=":")
    h_lim = [Line2D([0], [0], color=cor[t], lw=2.5, label=f"{t:g} mm")
             for t in limiares]
    h_mod = [Line2D([0], [0], color="black", linestyle=_estilo(m)[0],
                    marker=_estilo(m)[1], label=m) for m in modelos]
    fig.tight_layout(rect=[0, 0, 1, 0.85])   # faixa no topo p/ legenda+titulo
    fig.legend(handles=h_lim + h_mod, loc="upper left", bbox_to_anchor=(0.01, 0.945),
               fontsize=8, ncol=len(h_lim) + len(h_mod), framealpha=0.9,
               borderaxespad=0.3, columnspacing=1.0, handlelength=2.4)
    fig.suptitle(f"Metricas categoricas x prazo - janela {janela}  "
                 f"(cor=limiar; jaci continua, XC50 tracejada)", fontsize=12,
                 x=0.5, y=0.99)
    cam = os.path.join(saida, f"categoricas_{janela}.png")
    fig.savefig(cam, dpi=130); plt.close(fig); print(f"  {cam}")


def plota_fss(dff, janela, saida):
    modelos = sorted(dff.modelo.unique())
    limiares = sorted(dff.limiar_mm.unique())
    escalas = sorted(dff.escala_px.unique())
    cor = _cores_limiar(limiares)
    n = len(escalas); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow),
                            squeeze=False)
    for k, esc in enumerate(escalas):
        ax = axs.ravel()[k]
        for m in modelos:
            ls, mk = _estilo(m)
            for t in limiares:
                sub = dff[(dff.modelo == m) & (dff.limiar_mm == t) &
                          (dff.escala_px == esc)].sort_values("lead")
                if sub.empty:
                    continue
                ax.plot(sub.lead, sub.FSS, ls, marker=mk, color=cor[t],
                        markersize=4, lw=1.8)
        ax.set_title(f"FSS - vizinhanca {esc} px")
        ax.set_xlabel("Prazo (D+)"); ax.set_ylabel("FSS")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.set_xticks(sorted(dff.lead.unique()))
    for k in range(n, nrow * ncol):
        axs.ravel()[k].axis("off")
    h_lim = [Line2D([0], [0], color=cor[t], lw=2.5, label=f"{t:g} mm")
             for t in limiares]
    h_mod = [Line2D([0], [0], color="black", linestyle=_estilo(m)[0],
                    marker=_estilo(m)[1], label=m) for m in modelos]
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.legend(handles=h_lim + h_mod, loc="upper left", bbox_to_anchor=(0.01, 0.94),
               fontsize=8, ncol=len(h_lim) + len(h_mod), framealpha=0.9,
               borderaxespad=0.3, columnspacing=1.0, handlelength=2.4)
    fig.suptitle(f"FSS x prazo - janela {janela}  "
                 f"(cor=limiar; jaci continua, XC50 tracejada)", fontsize=12,
                 x=0.5, y=0.99)
    cam = os.path.join(saida, f"fss_{janela}.png")
    fig.savefig(cam, dpi=130); plt.close(fig); print(f"  {cam}")


# ------------------------------------------------------------------- mapas ---
def plota_mapas(grade, arrs, modelo, saida):
    lats = np.asarray(grade["lats"]); lons = np.asarray(grade["lons"])
    with np.errstate(invalid="ignore"):
        n = arrs["n"]
        mp = np.where(n > 0, arrs["sp"] / n, np.nan)
        mo = np.where(n > 0, arrs["so"] / n, np.nan)
        mb = np.where(n > 0, arrs["sdif"] / n, np.nan)
    if lats[0] > lats[-1]:          # norte->sul: inverte para origin='lower'
        mp, mo, mb = mp[::-1], mo[::-1], mb[::-1]
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    val = np.concatenate([mo[np.isfinite(mo)], mp[np.isfinite(mp)]])
    vmax = float(np.nanpercentile(val, 99)) if val.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    bvals = np.abs(mb[np.isfinite(mb)])
    bmax = float(np.nanpercentile(bvals, 99)) if bvals.size else 1.0
    if not np.isfinite(bmax) or bmax <= 0:
        bmax = 1.0
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for a, campo, tit, cmap, vlim in [
        (ax[0], mo, "Obs media (MERGE)", "Blues", (0, vmax)),
        (ax[1], mp, "Prev media", "Blues", (0, vmax)),
        (ax[2], mb, "Vies medio (Prev-Obs)", "RdBu_r", (-bmax, bmax)),
    ]:
        im = a.imshow(campo, origin="lower", extent=ext, aspect="auto",
                      cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        a.set_title(tit); a.set_xlabel("lon"); a.set_ylabel("lat")
        fig.colorbar(im, ax=a, shrink=0.85, label="mm")
    fig.suptitle(f"Verificacao 24h - modelo {modelo} (grade MERGE 10km)")
    fig.tight_layout()
    cam = os.path.join(saida, f"mapas_{modelo}.png")
    fig.savefig(cam, dpi=120); plt.close(fig); print(f"  {cam}")


# -------------------------------------------------------------------- main ---
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gera todas as figuras a partir do binario verificacao.pkl "
                    "(jaci continua, XC50 tracejada).")
    ap.add_argument("--binario", default="resultados_eta/verificacao.pkl",
                    help="caminho do verificacao.pkl salvo pelo verifica_eta.py")
    ap.add_argument("--figuras", default=None,
                    help="pasta de saida das figuras (padrao: a do binario)")
    ap.add_argument("--janelas", nargs="*", default=None,
                    help="janelas a plotar (ex.: 12Z 00Z); padrao: todas presentes")
    ap.add_argument("--sem-mapas", action="store_true")
    ap.add_argument("--sem-linhas", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.binario):
        print(f"ERRO: binario nao encontrado: {args.binario}\n"
              f"Rode antes: python verifica_eta.py --base ... --saida ...")
        return 1
    with open(args.binario, "rb") as f:
        b = pickle.load(f)
    saida = args.figuras or os.path.dirname(os.path.abspath(args.binario))
    os.makedirs(saida, exist_ok=True)

    tab = b.get("tabelas", {})
    dfp = tab.get("scores_prazo")
    dff = tab.get("scores_prazo_fss")

    if not args.sem_linhas and dfp is not None:
        janelas = args.janelas or sorted(dfp.janela.unique())
        for jan in janelas:
            print(f"Janela {jan}:")
            d = dfp[dfp.janela == jan]
            if d.empty:
                print(f"  (sem dados para {jan})"); continue
            plota_continuas(d, jan, saida)
            plota_categoricas(d, jan, saida)
            if dff is not None:
                dj = dff[dff.janela == jan]
                if not dj.empty:
                    plota_fss(dj, jan, saida)

    if not args.sem_mapas and b.get("mapas"):
        print("Mapas:")
        for modelo, arrs in b["mapas"].items():
            if np.asarray(arrs["n"]).sum() > 0:
                plota_mapas(b["grade"], arrs, modelo, saida)

    print(f"\nFiguras salvas em: {os.path.abspath(saida)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
