# ThermoBL-Net: Thermodynamic Boundary-Layer Sounder Network

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Target: JGR Atmospheres](https://img.shields.io/badge/Target-JGR%3A%20Atmospheres-blueviolet.svg)](https://agupubs.onlinelibrary.wiley.com/journal/21698996)

> **Thermodynamic Profiling of Pre-Monsoon Boundary-Layer Moisture Accumulation Using Hourly Geostationary Sounder Retrieval Networks**

---

## 1. Atmospheric Context & Scientific Motivation

During the pre-monsoon season (**March through May, MAM**), the Indian subcontinent experiences intense thermodynamic destabilization, resulting in severe localized convective storms, supercells, and squall lines—most notably the **Kalbaisakhi (Nor'westers)** over the Chhota Nagpur Plateau, Gangetic West Bengal, and Odisha, as well as severe thunderstorm clusters across the Deccan Plateau and the leeward foothills of the Western Ghats. These storms produce catastrophic microbursts, extreme lightning density, large hail, and localized flash flooding.

The convective lifecycle is governed by boundary-layer moisture convergence: warm, dry continental air from central/northwest India meets moist, maritime boundary-layer air advected from the northern Bay of Bengal or the Arabian Sea, establishing localized drylines, sea-breeze fronts, and sharp dew-point discontinuities. When surface solar insolation erodes the capping inversion (between 850 hPa and 700 hPa), Convective Available Potential Energy (CAPE) frequently exceeds $2500\text{--}4500\text{ J kg}^{-1}$, sparking rapid convective updrafts.

### The Observational Gap in Current Monitoring
| Platform | Capability | Critical Atmospheric Limitation |
| :--- | :--- | :--- |
| **Geostationary Imager** (INSAT-3DR/3DS Imager) | Cloud-top $T_B$, cooling rates | Diagnostic only ($T_B < 240\text{ K}$). Clouds have already breached the freezing level; misses 1–3 hr pre-convective moisture accumulation. |
| **Geostationary Sounder** (INSAT-3DR/3DS 19-Channel Sounder) | Hourly 40-level $T$ and $q$ profiles | Broad, overlapping weighting functions in lower troposphere (1000–850 hPa); high errors ($1.5\text{--}2.5\text{ K}$, $15\%\text{--}25\%\text{ RH}$) and dry bias. |
| **In-Situ Radiosondes** (IMD RS/RW) | High vertical fidelity | Coarse spatial spacing ($200\text{--}400\text{ km}$), twice daily (00/12 UTC); misses diurnal moisture convergence peaks (06:00–11:00 UTC). |
| **Ground GNSS Receivers** (SOI CORS / IMD) | Continuous 15-min PWV ($1\text{--}2\text{ mm}$) | Pure scalar column integral; zero information on vertical moisture stratification. |

**ThermoBL-Net** couples geostationary infrared sounder radiances with continuous ground geodetic GNSS networks, surface meteorological stations, and topography within a spatiotemporal deep learning architecture. The network downscales and reconstructs 3D volumetric specific humidity fields ($q(x, y, p, t)$) at standard boundary-layer pressure levels (**1000, 950, 925, 850, and 700 hPa**) over a 4 km grid, tracking moisture convergence zones and delivering **75 to 120 minutes of convective initiation lead time** before radar first-echo detection.

---

## 2. Multi-Sensor Input Architecture

| Dataset Name | Source / Provider | Spatial Resolution | Temporal Resolution | Channels / Variables | Role in Framework |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **INSAT-3DR/3DS Sounder** | ISRO MOSDAC | $10\text{ km}$ nadir ($0.04^\circ$ grid) | Hourly | 18 IR bands + 1 Visible band | Primary radiometric input & thermodynamic state |
| **Ground GNSS ZTD / PWV** | SOI CORS / IMD | $\approx 1000$ stations ($30\text{--}50\text{ km}$ spacing) | 15-minute | Integrated Precipitable Water Vapor ($\text{PWV}_{\text{GNSS}}$) | Physical column mass invariant constraint ($\mathcal{L}_{\text{column}}$) |
| **Radiosonde Soundings** | IMD Upper-Air Network | 14 primary storm-corridor stations | 00:00 & 12:00 UTC | $T, T_d, q, p, \mathbf{v}$ profiles | Ground-truth training supervision & verification |
| **ERA5 Atmospheric Reanalysis** | ECMWF / Copernicus | $0.25^\circ$ ($4\text{ km}$ downscaled) | Hourly | BLH, surface fluxes, winds ($u, v$) | Synoptic boundary-layer context & moisture flux operator |
| **INSAT-3DR Imager L2** | ISRO MOSDAC | $4\text{ km} \times 4\text{ km}$ | 30-minute | Land Surface Temperature (LST), Albedo | Diurnal boundary-layer heating driver |
| **SRTM Topography** | NASA JPL / USGS | $90\text{ m}$ (aggregated to $4\text{ km}$) | Static | Elevation ($h_{\text{srf}}$), Slope, Aspect, Curvature | Lower boundary pressure & orographic lifting corridors |
| **Doppler Weather Radar (DWR)** | IMD Radar Network | $1\text{ km} \times 1\text{ km}$ polar composites | 10- to 15-minute | Composite Reflectivity (MAX-Z) | Independent Convective Initiation ($\ge 35\text{ dBZ}$) validation |

---

## 3. Physical Formulation & Neural Architecture

### 3.1 GNSS Hydrostatic Mass Integral Operator
Ground GNSS Precipitable Water Vapor represents the vertical mass integral of specific humidity ($q(p)$):
$$\text{PWV} = \frac{1}{\rho_w g} \int_{p_{\text{top}}}^{p_{\text{srf}}} q(p) \, dp$$

In ThermoBL-Net, this integral acts as a physical boundary condition:
$$\mathcal{L}_{\text{column}} = \frac{1}{N_{\text{GNSS}}} \sum_{(x, y) \in \text{GNSS}} \left\vert \left( \frac{1}{\rho_w g} \sum_{k=1}^{L-1} \frac{\hat{q}(p_k) + \hat{q}(p_{k+1})}{2} \Delta p_k + \text{PWV}_{\text{upper}} \right) - \text{PWV}_{\text{GNSS}}(x, y) \right\vert$$

### 3.2 Spatiotemporal ConvLSTM Backbone
The model processes a rolling sequence of $K=4$ hours ($t-3, t-2, t-1, t$) with $C_{\text{in}} = 26$ input channels. The recurrent ConvLSTM cell replaces standard matrix multiplications with spatial convolutions, preserving spatial correlation structures (drylines, moisture surges, and orographic boundaries). The latent representation is decoded through a multi-scale transposed convolution network into 3D volumetric fields $\hat{\mathbf{Q}}_t \in \mathbb{R}^{5 \times H \times W}$ across $[1000, 950, 925, 850, 700]\text{ hPa}$.

### 3.3 Composite Physics-Informed Objective
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{profile}} + \lambda_2 \mathcal{L}_{\text{column}} + \lambda_3 \mathcal{L}_{\text{monotonic}} + \lambda_4 \mathcal{L}_{\text{conv}}$$
- $\mathcal{L}_{\text{profile}}$: Huber loss against collocated IMD radiosonde soundings.
- $\mathcal{L}_{\text{column}}$: Hydrostatic column mass conservation against GNSS PWV observations.
- $\mathcal{L}_{\text{monotonic}}$: Penalty against unphysical moisture lapse rates unless capping inversions are detected.
- $\mathcal{L}_{\text{conv}}$: Horizontal Moisture Flux Convergence ($\text{MFC} = -\nabla_h \cdot (\mathbf{v}_h \hat{q})$) evaluated via 2D finite-difference Sobel operators.

---

## 4. Verification Benchmark (Targeted at JGR: Atmospheres)

| Metric | Operational INSAT L2 | ERA5 Reanalysis | **ThermoBL-Net Target** | Scientific Significance |
| :--- | :--- | :--- | :--- | :--- |
| **$q$ RMSE at 925 hPa** | $2.8\text{--}3.4\text{ g kg}^{-1}$ | $1.8\text{--}2.2\text{ g kg}^{-1}$ | **$\le 1.1\text{ g kg}^{-1}$** | Resolves sharp dryline moisture gradients |
| **$q$ RMSE at 1000 hPa** | $3.2\text{--}4.1\text{ g kg}^{-1}$ | $2.1\text{--}2.6\text{ g kg}^{-1}$ | **$\le 1.3\text{ g kg}^{-1}$** | Captures maritime low-level moisture surges |
| **RH Bias ($< 850\text{ hPa}$)** | $-12\%\text{ to }-18\%$ | $+5\%\text{ to }+8\%$ | **$[-3.0\%, +3.0\%]$** | Eliminates satellite IR sounder boundary dry bias |
| **Total PWV RMSE (vs. GNSS)** | $4.2\text{--}5.5\text{ mm}$ | $2.8\text{--}3.5\text{ mm}$ | **$\le 1.6\text{ mm}$** | Reaches geodetic-grade water vapor precision |
| **CAPE Estimation RMSE** | $650\text{--}850\text{ J kg}^{-1}$ | $420\text{--}550\text{ J kg}^{-1}$ | **$\le 250\text{ J kg}^{-1}$** | Accurate severe storm thermodynamic energy |
| **Convective Initiation POD** | $0.41$ | $0.58$ | **$\ge 0.82$** | Pre-cloud storm anticipation |
| **Convective Initiation FAR** | $0.48$ | $0.39$ | **$\le 0.19$** | Suppresses anvil cirrus false alarms |
| **Convective Lead Time** | $15\text{--}30\text{ min}$ | $60\text{ min}$ | **$75\text{--}120\text{ min}$** | Proactive warning before radar $\ge 35\text{ dBZ}$ |

---

## 5. Installation & Setup

### Environment Setup
```bash
# Clone the repository
git clone <repository_url>
cd "Thermodynamic Profiling of Pre-Monsoon Boundary-Layer Moisture Accumulation Using Hourly Geostationary Sounder Retrieval Networks"

# Create conda environment
conda env create -f environment.yml
conda activate thermobl-net

# Alternatively, install via pip in an existing virtualenv
pip install -r requirements.txt
pip install -e .
```

### Running Unit Tests
```bash
pytest tests/ -v
```

---

## 6. Execution Workflow

### 1. Data Ingestion & Preprocessing
```bash
# Parse INSAT-3DR/3DS sounder HDF5, apply cloud mask, and reproject to 4 km grid
python src/data/insat_sounder_parser.py --config configs/data_config.yaml

# Compute GNSS PWV from ZTD and perform Voronoi-Kriging interpolation
python src/data/gnss_pwv_pipeline.py --config configs/data_config.yaml

# Collocate IMD Radiosondes
python src/data/radiosonde_collocation.py --config configs/data_config.yaml
```

### 2. Multi-GPU Distributed Training
```bash
# Launch PyTorch DistributedDataParallel (DDP) across 4 GPUs
torchrun --nproc_per_node=4 src/training/trainer.py --config configs/train_config.yaml
```

### 3. Ablation Studies
```bash
# Run the 4-way ablation suite (Sounder Only, +GNSS, +Topography/LST, Full ThermoBL-Net)
python src/training/ablation_runner.py --config configs/ablation_config.yaml
```

### 4. Convective Initiation Validation & Plotting
```bash
# Evaluate Convective Initiation (CI) lead time against IMD Doppler Weather Radar
python src/evaluation/ci_nowcasting.py --radar_dir data/raw/imd_radar --threshold 35.0

# Generate publication figures for JGR: Atmospheres
python src/visualization/publication_figures.py --checkpoint checkpoints/best_model.pt
```

---

## 7. Citation
If you use ThermoBL-Net or this codebase in your research, please cite:
```bibtex
@article{thermobonet2024,
  title={Thermodynamic Profiling of Pre-Monsoon Boundary-Layer Moisture Accumulation Using Hourly Geostationary Sounder Retrieval Networks},
  author={GeoVision Atmospheric AI Research Team},
  journal={Journal of Geophysical Research: Atmospheres},
  year={2024},
  note={Under Review}
}
```
