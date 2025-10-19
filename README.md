# 🎧 MagTapeDB: Dataset and Experiments for the Restoration of Historical Music Recordings

This repository accompanies the paper **"Diffusion-Based Denoising of Historical Recordings"** and gathers resources developed within the project **Restoration of Musicological Tape Recordings Using Deep Learning Models**, led by the **Audio Processing Group (GPA)** at the **Institute of Electrical Engineering, Universidad de la República (Uruguay)**, in collaboration with the **Lauro Ayestarán Center for Musical Documentation (CDM)**.

The repository includes:
- 🧩 **MagTapeDB** — a dataset of historical magnetic tape recordings, available under **Releases**.  
- 🔉 **Denoising experiments** — deep learning models and evaluation scripts for tape noise reduction.  
- ⚡ **Playback speed estimation experiments** — *Electric Network Frequency* (ENF)-based analysis for detecting and correcting playback speed variations.

---

## 🗂️ Repository Structure

```
MagTapeDB/
│
├── data/                     # Links or examples from the dataset
│   ├── examples/             # Demonstration audio fragments
│   └── metadata.csv          # Metadata (instrument, year, tape number, etc.)
│
├── denoising/                # Denoising experiments
│   ├── configs/              # Training and evaluation parameters
│   ├── models/               # Checkpoints and architectures (U-Net, Diffusion)
│   ├── scripts/              # Training and inference scripts
│   └── results/              # Processed audio examples
│
├── enf_speed_estimation/     # Playback speed estimation (ENF)
│   ├── analysis/             # ENF extraction and analysis scripts
│   ├── visualization/        # Plots and results
│   └── data/                 # Audio fragments used in ENF experiments
│
├── LICENSE
└── README.md
```

---

## 🎙️ MagTapeDB Dataset

**MagTapeDB** contains over **800 audio excerpts** derived from musicological tape recordings.  
It includes:

- Musical performances (field and studio recordings)  
- Isolated tape hiss segments  
- Pitchpipe tones used as tuning references  
- Detailed metadata (instrumentation, year, tape reel number, recording context)

The dataset is available in the [📦 Releases](../../releases) section of this repository.

---

## 🧠 Denoising Experiments

The denoising module reproduces and compares multiple noise reduction approaches, including:

- **U-Net** and **Diffusion-based** architectures  
- Supervised training on paired clean/noisy audio  
- Objective and perceptual evaluation metrics (SNR, SI-SDR, PESQ)

Example usage:

```bash
python denoising/train.py --config configs/unet_baseline.yaml
python denoising/evaluate.py --input data/examples/noisy.wav --model results/checkpoints/unet_latest.pth
```

---

## ⚙️ Playback Speed Estimation (ENF)

The **Electric Network Frequency (ENF)** component serves as a temporal reference for identifying playback speed fluctuations in analog recordings.

This module includes:
- ENF extraction from the audio signal  
- Comparison against historical ENF reference data  
- Playback rate estimation and correction

Example:

```bash
python enf_speed_estimation/estimate_enf.py --input data/examples/recording.wav
python enf_speed_estimation/correct_speed.py --input recording.wav --enf_data enf_track.npy
```

---

## 📊 Expected Results

| Task                      | Main Metrics         | Evaluated Models           |
|----------------------------|----------------------|-----------------------------|
| Denoising                 | SNR, SI-SDR, PESQ    | U-Net, Diffusion, NAM      |
| Playback speed estimation  | PPM error, Δtempo    | ENF tracking, FFT analysis |

These experiments reproduce the results discussed in the paper and serve as a reference framework for future research in audio restoration.

---

## 🧾 Citation

If you use this dataset or code, please cite:

```
Irigaray, I., Biscainho, L. W. P., et al. (2025).
Diffusion-Based Denoising of Historical Recordings.
Journal of the Audio Engineering Society (submitted).
```

---

## 🪶 License

The code is released under the **MIT License**.  
Audio samples and metadata in **MagTapeDB** are distributed under a **CC BY-NC-SA 4.0** license  
(*Attribution–NonCommercial–ShareAlike*).

---

## 🤝 Acknowledgments

- **Lauro Ayestarán Center for Musical Documentation (CDM)**  
- **Institute of Electrical Engineering – Universidad de la República (Uruguay)**  
- **Audio Processing Group (GPA)**  
- **Signal Processing Laboratory (LPS) – UFRJ**  
- CSIC Project: *Development of Tools for the Analysis of the Melodic Content of Audio Signals*
