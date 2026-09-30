# SCADA Pipeline Anomaly Detection

Project Deep Learning untuk mendeteksi kondisi normal/anomali pada operasi pipeline menggunakan **Multilayer Perceptron (MLP)** dari data telemetry SCADA.

## Dataset

Dataset `data/scada_pipeline.csv` berisi **1.000 baris dan 13 kolom**. Target `0 = Normal` dan `1 = Anomaly`.

### Fitur MLP

- pressure
- flow_rate
- temperature
- valve_status
- pump_state
- pump_speed
- compressor_state
- energy_consumption

`target` adalah label. `timestamp`, `segment_id`, dan `event_type` tidak dimasukkan sebagai fitur model. `alarm_triggered` juga tidak digunakan sebagai input utama agar model tidak bergantung pada indikator alarm.

## Struktur

```text
scada_pipeline_project/
├── api/
│   └── index.py                 # FastAPI untuk Vercel
├── data/
│   └── scada_pipeline.csv       # dataset
├── model/
│   ├── config.py                # fitur, threshold, profil tim
│   ├── inference.py             # inference tanpa PyTorch
│   ├── scaler.joblib            # dibuat setelah training
│   ├── mlp_weights.npz          # dibuat setelah training
│   └── metrics.json             # dibuat setelah training
├── outputs/
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── recent.json
├── public/
│   ├── index.html
│   └── assets/
│       ├── app.js
│       └── style.css
├── training/
│   └── train.py                # training MLP PyTorch
├── requirements.txt             # dependency deployment
├── requirements-training.txt    # dependency training lokal
├── vercel.json
└── README.md
```

## 1. Persiapan VS Code

Pastikan Python 3.12+ terpasang. Buka folder project di VS Code.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-training.txt
```

## 2. Training MLP

Dari root project:

```powershell
python training\train.py
```

Training menggunakan split stratified 80:20 untuk train/test dan validation split dari data train. Scaler hanya di-fit pada data train.

Arsitektur:

```text
8 input features
      ↓
Dense 32 + ReLU + Dropout 0.20
      ↓
Dense 16 + ReLU + Dropout 0.10
      ↓
Dense 8 + ReLU
      ↓
Dense 1 + Sigmoid
      ↓
Normal / Anomaly
```

Decision threshold output sigmoid = **0.50**. Ini threshold keputusan klasifikasi, bukan reconstruction-error threshold seperti Autoencoder.

## 3. Jalankan lokal

Install dependency runtime jika belum:

```powershell
python -m pip install -r requirements.txt
```

Jalankan FastAPI:

```powershell
uvicorn app:app --reload
```

Buka:

`http://127.0.0.1:8000`

API docs:

`http://127.0.0.1:8000/docs`

## 4. GitHub

Buat repository, misalnya:

`scada-pipeline-anomaly-detection`

```powershell
git init
git add .
git commit -m "Initial SCADA pipeline anomaly detection project"
git branch -M main
git remote add origin https://github.com/USERNAME/scada-pipeline-anomaly-detection.git
git push -u origin main
```

Sebelum push, ganti URL GitHub project dan URL LinkedIn/GitHub anggota pada `model/config.py`. Frontend akan mengambil data profil tersebut melalui API.

## 5. Deploy ke Vercel

Vercel saat ini mendukung FastAPI/Python. Static assets diletakkan di `public/` dan FastAPI diekspor melalui `app` instance. Vercel CLI dapat digunakan untuk deploy.

Install CLI:

```powershell
npm install -g vercel
```

Login:

```powershell
vercel login
```

Dari root project:

```powershell
vercel
```

Jika preview sudah benar:

```powershell
vercel --prod
```

Alternatifnya, import repository GitHub langsung dari dashboard Vercel. Set **Root Directory** ke root repository project ini dan jangan menjalankan training saat deployment.

## Catatan deployment

Training dilakukan **lokal**, bukan saat request Vercel. Deployment hanya membawa scaler + bobot MLP yang sudah dilatih dan menjalankan inference dengan NumPy. Ini membuat runtime jauh lebih ringan karena PyTorch hanya diperlukan saat training.

Jangan upload `.venv/`, cache, atau file training yang tidak diperlukan.

## Link profil

Edit daftar `TEAM` di `model/config.py` dan array `team` di `public/assets/app.js` dengan URL LinkedIn/GitHub asli masing-masing anggota.
