"""Генерация графиков распределений для reference/terver.

Запуск (NixOS, venv без libstdc++ в пути):
  LD_LIBRARY_PATH=<nix gcc lib> .venv/bin/python scripts/gen_distribution_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "reference" / "terver" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def stem(ax, ks, pmf, label, color):
    ax.vlines(ks, 0, pmf, color=color, lw=2)
    ax.plot(ks, pmf, "o", color=color, ms=5, label=label)


# ---------- дискретные ----------

def bernoulli():
    fig, ax = plt.subplots(figsize=(4, 2.8))
    stem(ax, [0, 1], [0.6, 0.4], "$p=0.4$", "tab:blue")
    stem(ax, [0, 1], [0.25, 0.75], "$p=0.75$", "tab:orange")
    ax.set_xticks([0, 1])
    ax.set_xlabel("$k$")
    ax.set_title("Бернулли: $P(X=1)=p$")
    ax.legend()
    save(fig, "bernoulli.png")


def binomial():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for n, p, c in [(10, 0.3, "tab:blue"), (10, 0.5, "tab:orange"), (50, 0.5, "tab:green")]:
        ks = np.arange(0, n + 1)
        pmf = stats.binom.pmf(ks, n, p)
        stem(ax, ks, pmf, f"n={n}, p={p}", c)
    ax.set_xlabel("$k$")
    ax.set_title("Биномиальное: $P(X=k)=C_n^k p^k (1-p)^{n-k}$")
    ax.legend()
    save(fig, "binomial.png")


def geometric():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for p, c in [(0.5, "tab:blue"), (0.2, "tab:orange")]:
        ks = np.arange(1, 16)
        pmf = stats.geom.pmf(ks, p)
        stem(ax, ks, pmf, f"p={p}", c)
    ax.set_xlabel("$k$ — номер первого успеха")
    ax.set_title("Геометрическое: $P(X=k)=(1-p)^{k-1}p$")
    ax.legend()
    save(fig, "geometric.png")


def poisson():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for lam, c in [(2, "tab:blue"), (5, "tab:orange"), (10, "tab:green")]:
        ks = np.arange(0, 25)
        pmf = stats.poisson.pmf(ks, lam)
        stem(ax, ks, pmf, f"$\\lambda$={lam}", c)
    ax.set_xlabel("$k$")
    ax.set_title("Пуассона: $P(X=k)=\\dfrac{\\lambda^k}{k!}e^{-\\lambda}$")
    ax.legend()
    save(fig, "poisson.png")


def hypergeometric():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    N, K, n = 30, 10, 7
    ks = np.arange(0, min(n, K) + 1)
    pmf = stats.hypergeom.pmf(ks, N, K, n)
    stem(ax, ks, pmf, f"N={N}, K={K}, n={n}", "tab:purple")
    ax.set_xlabel("$k$ — успехов в выборке")
    ax.set_title("Гипергеометрическое (выборка без возвращения)")
    ax.legend()
    save(fig, "hypergeometric.png")


def uniform_discrete():
    fig, ax = plt.subplots(figsize=(4, 2.8))
    ks = np.arange(1, 7)
    stem(ax, ks, np.full(6, 1 / 6), "кость", "tab:red")
    ax.set_xticks(ks)
    ax.set_title("Дискретное равномерное")
    ax.legend()
    save(fig, "uniform_discrete.png")


# ---------- непрерывные ----------

def uniform_cont():
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    x = np.linspace(-0.5, 1.5, 400)
    ax.plot(x, stats.uniform.pdf(x, 0, 1), color="tab:blue")
    ax.set_title("Непрерывное равномерное на $[0,1]$")
    save(fig, "uniform.png")


def normal():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.linspace(-6, 6, 400)
    for mu, sigma, c in [(0, 1, "tab:blue"), (2, 0.7, "tab:orange"), (0, 2, "tab:green")]:
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color=c,
                label=f"$\\mu$={mu}, $\\sigma$={sigma}")
    ax.set_title("Нормальное")
    ax.legend()
    save(fig, "normal.png")


def exponential():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.linspace(0, 6, 400)
    for lam, c in [(1, "tab:blue"), (0.5, "tab:orange"), (2, "tab:green")]:
        ax.plot(x, stats.expon.pdf(x, scale=1 / lam), color=c, label=f"$\\lambda$={lam}")
    ax.set_title("Экспоненциальное: $f(x)=\\lambda e^{-\\lambda x}$")
    ax.legend()
    save(fig, "exponential.png")


def gamma_dist():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.linspace(0, 15, 500)
    for k, theta, c in [(2, 1, "tab:blue"), (1, 2, "tab:orange"), (5, 1, "tab:green"), (3, 2, "tab:red")]:
        ax.plot(x, stats.gamma.pdf(x, k, scale=theta), color=c, label=f"k={k}, $\\theta$={theta}")
    ax.set_title("Гамма: $k$-я форма, $\\theta$-й масштаб")
    ax.legend()
    save(fig, "gamma.png")


def beta_dist():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.linspace(0.001, 0.999, 400)
    cases = [(1, 1, "табличка ~ равномерное"), (2, 2, "колокол"),
             (5, 2, "скошено к 1"), (0.5, 0.5, "U-образная")]
    for (a, b, label), c in zip(cases, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        ax.plot(x, stats.beta.pdf(x, a, b), color=c, label=f"$\\alpha$={a}, $\\beta$={b} ({label})")
    ax.set_title("Бета")
    ax.legend()
    save(fig, "beta.png")


def dirichlet():
    """Плотность на 2-симплексе (треугольник) для трёх наборов параметров."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    grid = np.linspace(0, 1, 500)
    X, Y = np.meshgrid(grid, grid)
    Z3 = 1 - X - Y
    mask = Z3 >= -1e-12
    Xc, Yc, Zc = (np.clip(X[mask], 1e-12, None),
                  np.clip(Y[mask], 1e-12, None),
                  np.clip(Z3[mask], 1e-12, None))

    cases = [((1, 1, 1), "равномерный на симплексе"),
             ((10, 10, 10), "концентрированный к центру"),
             ((2, 3, 4), "скошен к $\\theta_3$")]
    for ax, (alpha, title) in zip(axes, cases):
        logf = (alpha[0] - 1) * np.log(Xc) + (alpha[1] - 1) * np.log(Yc) \
            + (alpha[2] - 1) * np.log(Zc)
        logf -= logf.max()
        dens = np.exp(logf)
        D = np.full_like(X, np.nan)
        D[mask] = dens
        im = ax.imshow(D, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color="black", lw=1)
        ax.text(-0.03, -0.05, "$\\theta_2$", ha="right", va="top", fontsize=12)
        ax.text(1.03, -0.05, "$\\theta_3$", ha="left", va="top", fontsize=12)
        ax.text(-0.03, 1.02, "$\\theta_1$", ha="right", va="bottom", fontsize=12)
        ax.set_title(f"$\\alpha=({alpha[0]},{alpha[1]},{alpha[2]})$\n{title}")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Дирихле: плотность на треугольнике $(\\theta_1, \\theta_2, \\theta_3)$, $\\theta_i \\geq 0$, $\\sum \\theta_i = 1$",
                 y=1.06)
    save(fig, "dirichlet.png")


if __name__ == "__main__":
    bernoulli()
    binomial()
    geometric()
    poisson()
    hypergeometric()
    uniform_discrete()
    uniform_cont()
    normal()
    exponential()
    gamma_dist()
    beta_dist()
    dirichlet()
