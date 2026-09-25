# MonsoonIQ Weather Regime Classification Guide

## Overview
Precipitation post-processing in India cannot assume a stationary, spatially uniform bias structure. Physical rainfall generation mechanisms vary drastically across synoptic regimes. MonsoonIQ introduces 7 meteorologically classified regimes grounded in Indian Meteorological Department (IMD) synoptic climatology.

---

## The 7 Regimes and Their Signatures

| Regime ID | Regime Name | Meteorological Definition | Key Predictor Signatures | NWP Bias Mechanism |
|---|---|---|---|---|
| **1** | **Active Monsoon** | Vigorous southwest monsoon flow; monsoon trough located over central India ($18^\circ-26^\circ\text{N}$). | $U_{850} \ge 12\text{ m/s}$, Cyclonic Vorticity $\ge 1.5\times 10^{-5}\text{ s}^{-1}$, OLR anomaly $\le -15\text{ W/m}^2$. | Spatial misplacement along trough; intensity underestimation in convective cores. |
| **2** | **Break Monsoon** | Sea-level monsoon trough shifts to the foothills of the Himalayas ($27^\circ-32^\circ\text{N}$); suppressed rain over central India. | Trough lat $\ge 27^\circ\text{N}$, Central OLR anomaly $\ge +12\text{ W/m}^2$, $U_{850} \le 7\text{ m/s}$. | NWP maintains spurious central convection, creating severe wet bias. |
| **3** | **Monsoon Low / Depression** | Synoptic cyclonic vortex originating in the Bay of Bengal ($996\text{ hPa}$ central minimum). | 850 hPa Vorticity $\ge 4.0\times 10^{-5}\text{ s}^{-1}$, MSLP anomaly $\le -3.5\text{ hPa}$, Moisture flux conv $\ge 3\times 10^{-4}\text{ kg/(m}^2\text{ s)}$. | Position displacement of 100–200 km; dipole errors with intense missed rain in SW quadrant. |
| **4** | **Orographic** | Mechanical lifting of moist maritime flow against Western Ghats and southern Himalayan slopes. | Elevation $\ge 350\text{ m}$, Terrain slope $\ge 0.012$, Onshore wind $\ge 8\text{ m/s}$, $\text{RH}_{850} \ge 80\%$. | Coarse model grid (27 km) smooths ridge height, underestimating rainfall by 40–50%. |
| **5** | **Coastal** | Differential land-sea friction and sea-breeze convergence within 65 km of coastlines. | Distance to coast $\le 65\text{ km}$, Coastal convergence $\ge 1.5\times 10^{-5}\text{ s}^{-1}$, marine moisture flux $\ge 250\text{ kg/(m s)}$. | Unresolved microscale coastal front produces chronic dry bias. |
| **6** | **Western Disturbance** | Subtropical westerly upper-level trough moving eastward over NW India ($25^\circ-36^\circ\text{N}$, $\le 82^\circ\text{E}$). | 500 hPa Geopotential trough anomaly $\le -35\text{ gpm}$, 500 hPa Vorticity $\ge 2.0\times 10^{-5}\text{ s}^{-1}$ (Oct–May). | Missed orographic cloudbursts in deep valleys; over-smoothed widespread light rain. |
| **7** | **Weak / Normal** | Baseline background convection without extreme synoptic or orographic forcing. | Climatological background state. | Mild convective scale error. |

---

## Soft Blending Equation
To prevent discontinuous "step function" jumps at geographic and temporal regime boundaries, predictions are soft-blended:

$$\hat{Y}_{\text{MonsoonIQ}} = \sum_{k=1}^7 P(R_k) \cdot \left[ Y_{QM, k} + \hat{\mathcal{R}}_k(X) \right]$$

Where:
- $P(R_k)$ is the calibrated soft probability of regime $k$ from the multi-class LightGBM classifier.
- $Y_{QM, k}$ is the empirical quantile mapping baseline for regime $k$.
- $\hat{\mathcal{R}}_k(X)$ is the LightGBM residual learning expert for regime $k$.
