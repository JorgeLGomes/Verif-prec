#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plota_metricas.py
=================

Gera os graficos de TODAS as metricas de verificacao em funcao do prazo (D+1,
D+2, ...), lendo os CSVs produzidos pelo verifica_eta.py.

Convencao de estilo (pedido do usuario):
    - jaci  -> linha CONTINUA  ('-')
    - XC50  -> linha TRACEJADA ('--')

Para cada janela (12Z, 00Z) sao gerados:
    - continuas_<janela>.png ..... vies, MAE, RMSE, correlacao (e vies mult.)
    - categoricas_<janela>.png ... POD, FAR, CSI, ETS, HSS, FBIAS (1 cor/limiar)
    - fss_<janela>.png ........... FSS por escala de vizinhanca (1 cor/limiar)

Uso:
    python plota_metricas.py --saida resultados_eta
    python plota_metricas.py --saida resultados_eta --janelas 12Z 00Z

Le: scores_por_prazo.csv e scores_por_prazo_fss.csv (na pasta --saida).
Depende de: pandas, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# estilo de linha por modelo
ESTILO = {"jaci": "-", "xc50": "--"}
MARCA = {"jaci": "o", "xc50": "s"}


def _estilo(modelo):
    return ESTILO.get(modelo, "-"), MARCA.get(modelo, "o")


def _cores_limiar(limiares):
    """Cores de alto contraste por limiar (qualitativa forte ate 8; turbo acima)."""
    n = len(limiares)
    if n <= 8:
        base = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                "#17becf", "#8c564b", "#000000"]
        return {t: base[i % len(base)] for i, t in enumerate(limiares)}
    cores = plt.cm.turbo(np.linspace(0.02, 0.98, n))
    return {t: c for t, c in zip(limiares, cores)}


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
        for m in modelos:
            sub = d[d.modelo == m].sort_values("lead")
            ls, mk = _estilo(m)
            ax.plot(sub.lead, sub[col], ls, marker=mk, color=cor[m],
                    label=m, markersize=5, lw=1.8)
        ax.set_title(rot); ax.set_xlabel("Prazo (D+)"); ax.grid(alpha=0.3)
        ax.set_xticks(sorted(d.lead.unique()))
        if col in ("cont_bias",):
            ax.axhline(0, color="grey", lw=0.8, ls=":")
        if col == "cont_vies_mult":
            ax.axhline(1, color="grey", lw=0.8, ls=":")
    h = [Line2D([0], [0], color="black", linestyle=_estilo(m)[0],
                marker=_estilo(m)[1],
                label=f"{m} ({'continua' if _estilo(m)[0]=='-' else 'tracejada'})")
         for m in modelos]
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.985),
               fontsize=9, ncol=len(h), framealpha=0.9, borderaxespad=0.3)
    fig.suptitle(f"Metricas continuas x prazo - janela {janela}", fontsize=13,
                 x=0.5, y=0.995)
    cam = os.path.join(saida, f"continuas_{janela}.png")
    fig.savefig(cam, dpi=130); plt.close(fig)
    print(f"  {cam}")


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

    # legenda dupla (cor=limiar, estilo=modelo) no topo-esquerdo da figura
    h_lim = [Line2D([0], [0], color=cor[t], lw=2.5, label=f"{t:g} mm")
             for t in limiares]
    h_mod = [Line2D([0], [0], color="black", linestyle=_estilo(m)[0],
                    marker=_estilo(m)[1], label=m) for m in modelos]
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.legend(handles=h_lim + h_mod, loc="upper left", bbox_to_anchor=(0.01, 0.945),
               fontsize=8, ncol=len(h_lim) + len(h_mod), framealpha=0.9,
               borderaxespad=0.3, columnspacing=1.0, handlelength=2.4)
    fig.suptitle(f"Metricas categoricas x prazo - janela {janela}  "
                 f"(cor=limiar; jaci continua, XC50 tracejada)", fontsize=12,
                 x=0.5, y=0.99)
    cam = os.path.join(saida, f"categoricas_{janela}.png")
    fig.savefig(cam, dpi=130); plt.close(fig)
    print(f"  {cam}")


def plota_fss(dff, janela, saida):
    modelos = sorted(dff.modelo.unique())
    limiares = sorted(dff.limiar_mm.unique())
    escalas = sorted(dff.escala_px.unique())
    cor = _cores_limiar(limiares)
    n = len(escalas)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))
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
    fig.savefig(cam, dpi=130); plt.close(fig)
    print(f"  {cam}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Graficos das metricas x prazo (jaci continua, XC50 tracejada).")
    ap.add_argument("--saida", default="resultados_eta",
                    help="pasta com os CSVs do verifica_eta.py")
    ap.add_argument("--janelas", nargs="*", default=None,
                    help="janelas a plotar (ex.: 12Z 00Z); padrao: todas presentes")
    args = ap.parse_args(argv)

    fprazo = os.path.join(args.saida, "scores_por_prazo.csv")
    ffss = os.path.join(args.saida, "scores_por_prazo_fss.csv")
    if not os.path.isfile(fprazo):
        print(f"ERRO: nao achei {fprazo}. Rode o verifica_eta.py antes.")
        return 1
    df = pd.read_csv(fprazo)
    dff = pd.read_csv(ffss) if os.path.isfile(ffss) else None

    janelas = args.janelas or sorted(df.janela.unique())
    for jan in janelas:
        print(f"Janela {jan}:")
        d = df[df.janela == jan]
        if d.empty:
            print(f"  (sem dados para {jan})"); continue
        plota_continuas(d, jan, args.saida)
        plota_categoricas(d, jan, args.saida)
        if dff is not None:
            dfj = dff[dff.janela == jan]
            if not dfj.empty:
                plota_fss(dfj, jan, args.saida)
    print(f"\nGraficos salvos em: {os.path.abspath(args.saida)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
