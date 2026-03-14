"""
Simulation code for:
  Can LLM Agents Care About the World?
  Exocentric Homeostatic Deliberation (EHD)

Author : Luca Lillo, University of Liverpool, 2026
Figures: Propositions 4, 5 and Section 7.
Usage  : python ehd_simulations.py
Output : figures/fig_convergence.{pdf,png}
         figures/fig_ranking_divergence.{pdf,png}
         figures/fig_welfare_trajectory.{pdf,png}
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

np.random.seed(42)
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

NAVY  = "#1B3A6B"
PLUM  = "#7B2D8B"
GREEN = "#2E7D32"
GOLD  = "#DAA520"
RED   = "#B71C1C"
GRAY  = "#757575"


def generate_fig1():
    mu_star, mu_0, sigma_obs = 0.35, 0.60, 0.05
    K, N = 200, 50
    trajectories = np.zeros((N, K + 1))
    for n in range(N):
        mu = mu_0
        trajectories[n, 0] = mu
        for k in range(1, K + 1):
            kappa = 0.5 / (k ** 0.75)
            obs = np.random.normal(mu_star, sigma_obs)
            mu = mu + kappa * (obs - mu)
            trajectories[n, k] = mu

    ks = np.arange(K + 1)
    mean_tr = trajectories.mean(axis=0)
    p10 = np.percentile(trajectories, 10, axis=0)
    p90 = np.percentile(trajectories, 90, axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n in range(N):
        ax.plot(ks, trajectories[n], color=NAVY, alpha=0.12, lw=0.7)
    ax.fill_between(ks, p10, p90, color=NAVY, alpha=0.18)
    ax.plot(ks, mean_tr, color=NAVY, lw=2.0, label="Mean trajectory")
    ax.axhline(mu_star, color=GREEN, lw=1.6, ls="--",
               label=r"True mean $\mu^*(a)=0.35$")
    ax.axhline(mu_0, color=RED, lw=1.2, ls="--", alpha=0.6,
               label=r"Initial estimate $\mu_{\theta_0}=0.60$")
    ax.set_xlabel(r"Visit count $k$")
    ax.set_ylabel(r"Mean estimate $\mu_{\theta_k}(a)$")
    ax.set_title("Convergence of recalibration rule (Proposition 4)")
    legend_elements = [
        Line2D([0], [0], color=NAVY, lw=2.0,
               label="Mean trajectory"),
        Patch(facecolor=NAVY, alpha=0.3,
              label="10th–90th percentile"),
        Line2D([0], [0], color=GREEN, lw=1.6,
               ls="--",
               label=r"True mean $\mu^*(a)=0.35$"),
        Line2D([0], [0], color=RED, lw=1.2,
               ls="--", alpha=0.6,
               label=r"Initial estimate $\mu_{\theta_0}=0.60$"),
    ]
    ax.legend(handles=legend_elements,
              fontsize=9, frameon=False,
              loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/fig_convergence.pdf", dpi=300)
    fig.savefig("figures/fig_convergence.png", dpi=300)
    plt.close(fig)
    print("fig_convergence saved.")


def generate_fig2():
    mu_star, sigma_star = 0.50, 0.03
    sigma2_a = 0.02
    mu_a = mu_b = 0.45
    sigma2_b_vals = np.linspace(0.02, 0.15, 100)
    delta_sigma2 = sigma2_b_vals - sigma2_a

    def kl_gauss(mu_q, s2_q, mu_p, s2_p):
        return 0.5 * ((mu_q-mu_p)**2/s2_p + s2_q/s2_p
                      - 1 - np.log(s2_q/s2_p))

    kl_a = kl_gauss(mu_a, sigma2_a, mu_star, sigma_star**2)
    kl_b = kl_gauss(mu_b, sigma2_b_vals, mu_star, sigma_star**2)
    delta_kl = kl_a - kl_b
    delta_ehd = np.zeros_like(delta_sigma2)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color=GRAY, lw=0.8, ls=":")
    ax.fill_between(delta_sigma2, delta_ehd, delta_kl,
                    color=PLUM, alpha=0.08)
    ax.plot(delta_sigma2, delta_ehd, color=NAVY, lw=2.2,
            label=r"EHD $\Delta V_{\mathrm{prag}}=0$")
    ax.plot(delta_sigma2, delta_kl, color=PLUM, lw=2.2,
            label=r"EFE $\Delta\mathrm{KL}$")
    ax.text(0.003,  0.008,
            "EHD: indifferent to variance",
            fontsize=9, color=NAVY,
            va='bottom')
    ax.text(0.003, delta_kl[2] - 0.015,
            "EFE: penalises higher variance",
            fontsize=9, color=PLUM,
            va='top')
    ax.set_xlabel(
        r"Variance gap $\sigma^2(b)-\sigma^2(a)$")
    ax.set_ylabel(r"Score difference $(a)-(b)$")
    ax.set_title(
        "Ranking divergence: EHD vs single-step EFE\n"
        "(Proposition 5, mechanism ii)")
    ax.legend(fontsize=9, frameon=True,
              loc="lower left",
              framealpha=0.9,
              edgecolor="#e0e0e0")
    fig.tight_layout()
    fig.savefig("figures/fig_ranking_divergence.pdf", dpi=300)
    fig.savefig("figures/fig_ranking_divergence.png", dpi=300)
    plt.close(fig)
    print("fig_ranking_divergence saved.")


def generate_fig3():
    t = np.arange(0, 25, dtype=float)
    noise = np.random.normal(0, 0.01, len(t))
    g_raw = 0.08 * np.sin(2 * np.pi * t / 24) + 0.04 + noise
    g_smooth = np.zeros_like(g_raw)
    g_smooth[0] = g_raw[0]
    for i in range(1, len(t)):
        g_smooth[i] = 0.3 * g_raw[i] + 0.7 * g_smooth[i-1]
    W_ext = 1 / (1 + np.exp(3 * g_smooth))
    theta_ext = 0.45

    B = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < 3:
            B[i] = W_ext[i]
        elif ti == 3:
            B[i] = 0.62
        elif ti < 16:
            B[i] = 0.62 * np.exp(-0.08 * (ti - 3))
        else:
            B[i] = W_ext[i] + 0.03

    x2 = 0.80 * np.exp(-0.02 * t)
    x2[6:] += 0.12 * np.exp(-0.02 * (t[6:] - 6))
    x2 = np.clip(x2, 0, 1)
    x3 = 0.72 * np.exp(-0.01 * t)
    x1 = 0.70 * W_ext + 0.20 * B + 0.10 * 0.42
    V  = 0.55 * x1 + 0.25 * x2 + 0.20 * x3

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7),
                             sharex=True,
                             gridspec_kw={"height_ratios": [2.5,1,1]})
    fig.suptitle(
        "EHD welfare trajectory — 24-month simulation (Section 7)",
        fontsize=11)

    ax = axes[0]
    ax.plot(t, W_ext, color=PLUM, lw=2,
            label=r"$W_{\mathrm{ext}}(t)$")
    ax.plot(t, B, color=GOLD, lw=1.8, ls="--",
            label=r"$B_t(a^*)$")
    ax.axhline(theta_ext, color=GRAY, lw=1, ls=":",
               label=r"$\theta_{\mathrm{ext}}=0.45$")
    ax.fill_between(t, 0, 1, where=(W_ext < theta_ext),
                    color=RED, alpha=0.10, label="Trigger active")
    ax.annotate("Action\nselected",
                xy=(3, W_ext[3]),
                xytext=(4.5, 0.20),
                fontsize=8, color=GREEN,
                arrowprops=dict(
                    arrowstyle="->",
                    color=GREEN, lw=1.4,
                    connectionstyle="arc3,rad=0.2"))
    ax.set_ylabel("Welfare signal")
    ax.set_ylim(0.10, 0.85)
    ax.legend(fontsize=8, frameon=True,
              loc="upper left",
              framealpha=0.92,
              edgecolor="#e0e0e0",
              ncol=2,
              bbox_to_anchor=(0.0, 1.0))

    ax = axes[1]
    ax.plot(t, x2, color=NAVY, lw=1.8,
            label=r"Budget $x_{t,2}$")
    ax.plot(t, x3, color=GREEN, lw=1.8, ls="--",
            label=r"Knowledge $x_{t,3}$")
    ax.plot(6, x2[6], marker="^", color=NAVY, ms=7)
    ax.set_ylabel("Endocentric")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    ax.plot(t, V, color="black", lw=2, label=r"$V(x_t)$")
    ax.axhline(0.50, color=GRAY, lw=1, ls=":",
               label=r"$\theta=0.50$")
    ax.set_ylabel(r"$V(x_t)$")
    ax.set_xlabel(r"Time $t$ (months)")
    ax.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig("figures/fig_welfare_trajectory.pdf", dpi=300)
    fig.savefig("figures/fig_welfare_trajectory.png", dpi=300)
    plt.close(fig)
    print("fig_welfare_trajectory saved.")


def main():
    print("Generating EHD simulation figures...")
    generate_fig1()
    generate_fig2()
    generate_fig3()
    print("Done. Files saved in ./figures/")

if __name__ == "__main__":
    main()
