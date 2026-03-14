# EHD Simulations — Can LLM Agents Care About the World?

<img src="https://img.shields.io/badge/python-3.10%2B-blue">

Reproducible simulation code for the paper:

> **Can LLM Agents Care About the World? World-Directed Welfare
> and Exocentric Homeostatic Deliberation**
> Luca Lillo — University of Liverpool, 2026
> *arXiv preprint — under submission*

Paper evidence memo: https://www.neuromorphicinference.com/evidence/

## Figures produced

| File | Description |
|------|-------------|
| `figures/fig_convergence` | Proposition 4 — Robbins–Monro convergence. 50 trajectories converging to μ*(a)=0.35 from initial 0.60. |
| `figures/fig_ranking_divergence` | Proposition 5(ii) — EHD vs single-step EFE ranking divergence under equal means, differing variances. |
| `figures/fig_welfare_trajectory` | Section 7 — 24-month AMR monitoring simulation. |

## Usage

```
pip install -r requirements.txt
python ehd_simulations.py
```

Figures saved as PDF and PNG in `./figures/`

## Citation

```
@misc{lillo2026ehd,
  author = {Luca Lillo},
  title  = {Can {LLM} Agents Care About the World?},
  year   = {2026},
  note   = {arXiv preprint, under submission}
}
```

## Licence
MIT