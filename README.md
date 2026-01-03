# ECG Arrhythmia Classification using CNN-LSTM

Deep learning model for automated ECG classification detecting Normal Sinus Rhythm (NSR), Premature Atrial Contractions (PAC), and Premature Ventricular Contractions (PVC).

**Publication:**  
Bashar S.K. et al., 2026, "Premature atrial and ventricular contraction detection using deep learning and short ECG: A multi-dataset evaluation", *Biomedical Signal Processing and Control*, 113, p.108956.  
https://doi.org/10.1016/j.bspc.2025.108956

---

## 🚀 Quick Start with Docker

### Option 1: Use Pre-built Image (Recommended - Faster!)

```bash
docker pull skbashar09/ecg-classifier-bspc:latest

# Run predictions on test datasets
docker run skbashar09/ecg-classifier-bspc:latest --dataset cpsc_nsr
docker run skbashar09/ecg-classifier-bspc:latest --dataset cpsc_pac
docker run skbashar09/ecg-classifier-bspc:latest --dataset cpsc_pvc
docker run skbashar09/ecg-classifier-bspc:latest --dataset wearable_pvc
```


### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/skbashar09/ecg-pac-pvc-model.git
cd ecg-pac-pvc-model

# Build Docker image
docker build -t ecg-classifier-bspc .

# Run predictions on test datasets
docker run ecg-classifier-bspc --dataset cpsc_nsr
docker run ecg-classifier-bspc --dataset cpsc_pac
docker run ecg-classifier-bspc --dataset cpsc_pvc
docker run ecg-classifier-bspc --dataset wearable_pvc
```
---

## 📊 Results

| Dataset | Sample Segments | NSR | PAC | PVC | Other |
|---------|--------------|-----|-----|-----|-------|
| CPSC NSR | 212 | 209 | 3 | 0 | 0 |
| CPSC PAC | 204 | 24 | 178 | 1 | 1 |
| CPSC PVC | 111 | 5 | 11 | 93 | 2 |
| Wearable PVC | 103 | 0 | 0 | 103 | 0 |

---

## 🛠️ Technologies

- **Python 3.11**
- **PyTorch 2.5** (CPU-optimized)
- **Docker** for reproducibility
- **CNN-LSTM** architecture

---

## 📄 Citation

If you use this code, please cite:

```bibtex
@article{bashar2026ecg,
  title={Premature atrial and ventricular contraction detection using deep learning and short ECG: A multi-dataset evaluation},
  author={Bashar, Syed Khairul and Socia, Damien and Feuerwerker, Solomon and Dodson, Emmanuel and An, Gary and Cockrell, R Chase},
  journal={Biomedical Signal Processing and Control},
  volume={113},
  pages={108956},
  year={2026},
  publisher={Elsevier}
}
```

---

## 👤 Author


**Syed Khairul Bashar**  
PhD in Biomedical Engineering 
📧 skbashar09@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/syed-khairul-bashar/)
---
## 📝 License


MIT License - Free to use with attribution.
