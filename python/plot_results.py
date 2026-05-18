#!/usr/bin/env python3
"""Generate the benchmark charts from the JSON result files.

Reads results/benchmark.json, results/convergence.json, and results/surface.json
and writes four PNG charts to results/. The color system is a small validated
categorical palette so the charts read cleanly and stay colorblind safe. CPU is
blue and GPU is green throughout for consistency.

Charts produced:
  results/speedup.png       CPU vs GPU time across path counts (log-log)
  results/convergence.png   MC price converging to Black Scholes with error band
  results/surface.png       Call price heatmap over spot and volatility
  results/greeks.png        Monte Carlo vs analytical Greeks
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Validated categorical palette. Identity is carried by these two hues only.
CPU_COLOR = "#2a78d6"    # blue
GPU_COLOR = "#008300"    # green
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e7e7e4"


def _style_axes(ax):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  skip, missing {name}")
        return None
    with open(path) as f:
        return json.load(f)


def plot_speedup(bench):
    rows = [r for r in bench["scaling"] if r["cpu_ms"] is not None]
    if not rows:
        return
    paths = [r["paths"] for r in rows]
    cpu = [r["cpu_ms"] for r in rows]
    gpu = [r["gpu_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _style_axes(ax)
    ax.plot(paths, cpu, color=CPU_COLOR, linewidth=2, marker="o", markersize=6,
            label="CPU (NumPy)", zorder=3)
    ax.plot(paths, gpu, color=GPU_COLOR, linewidth=2, marker="o", markersize=6,
            label=f"GPU ({bench['meta']['device']})", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Monte Carlo paths", color=INK)
    ax.set_ylabel("Time (ms)", color=INK)
    ax.set_title("Pricing time, CPU vs GPU", color=INK, fontsize=13, loc="left")

    # Direct label the peak speedup rather than annotating every point.
    best = max(rows, key=lambda r: (r["speedup"] or 0))
    ax.annotate(f"{best['speedup']:.0f}x faster",
                xy=(best["paths"], best["gpu_ms"]),
                xytext=(best["paths"], best["gpu_ms"] * 4),
                color=GPU_COLOR, fontsize=11, fontweight="bold", ha="center")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "speedup.png"), dpi=140)
    plt.close(fig)
    print("  wrote speedup.png")


def plot_convergence(conv):
    rows = conv["rows"]
    paths = np.array([r["paths"] for r in rows])
    price = np.array([r["price"] for r in rows])
    se = np.array([r["std_error"] for r in rows])
    bs = conv["bs_price"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _style_axes(ax)
    ax.axhline(bs, color=MUTED, linewidth=1.5, linestyle="--", zorder=2,
               label="Black Scholes")
    ax.fill_between(paths, price - 1.96 * se, price + 1.96 * se, color=GPU_COLOR,
                    alpha=0.15, zorder=1)
    ax.plot(paths, price, color=GPU_COLOR, linewidth=2, marker="o", markersize=6,
            zorder=3, label="Monte Carlo")
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo paths", color=INK)
    ax.set_ylabel("Call price", color=INK)
    ax.set_title(f"Convergence to analytical price (slope {conv['slope']:.3f})",
                 color=INK, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "convergence.png"), dpi=140)
    plt.close(fig)
    print("  wrote convergence.png")


def plot_surface(surf):
    spots = np.array(surf["spots"])
    vols = np.array(surf["vols"])
    prices = np.array(surf["prices"])

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    # Magnitude data, so a single hue sequential ramp, light to dark.
    mesh = ax.pcolormesh(spots, vols, prices, cmap="Blues", shading="auto")
    ax.set_xlabel("Spot price", color=INK)
    ax.set_ylabel("Volatility", color=INK)
    ax.set_title(f"European call price surface (strike {surf['strike']:.0f})",
                 color=INK, fontsize=13, loc="left")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Price", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "surface.png"), dpi=140)
    plt.close(fig)
    print("  wrote surface.png")


def plot_greeks(bench):
    g = bench["greeks"]
    names = ["delta", "gamma", "vega", "theta"]
    mc = [g["monte_carlo"][n] for n in names]
    an = [g["analytical"][n] for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.6))
    for ax, name, m, a in zip(axes, names, mc, an):
        _style_axes(ax)
        ax.bar([0], [m], width=0.6, color=GPU_COLOR, label="Monte Carlo", zorder=3)
        ax.bar([1], [a], width=0.6, color=CPU_COLOR, label="Analytical", zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MC", "BS"], color=MUTED)
        ax.set_title(name.capitalize(), color=INK, fontsize=12)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Greeks, Monte Carlo vs analytical", color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(RESULTS_DIR, "greeks.png"), dpi=140)
    plt.close(fig)
    print("  wrote greeks.png")


def main():
    print("Generating charts")
    bench = load("benchmark.json")
    conv = load("convergence.json")
    surf = load("surface.json")
    if bench:
        plot_speedup(bench)
        plot_greeks(bench)
    if conv:
        plot_convergence(conv)
    if surf:
        plot_surface(surf)
    print("Done")


if __name__ == "__main__":
    main()
