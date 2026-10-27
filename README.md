# Fashion Image Generator — DCGAN
### DSA Class Test · Task 4

Generates new clothing images using a **Deep Convolutional GAN** trained on Fashion-MNIST.

---

## Common Information Checklist

| Requirement | Implementation |
|---|---|
| ✅ **Frontend** | Streamlit app with 6 meaningful controls (image upload, model selector, epoch slider, batch size, temperature, seed) |
| ✅ **Backend** | Separate modules: `model.py`, `preprocessing.py`, `train.py`, `inference.py` |
| ✅ **Actual model execution** | PyTorch DCGAN — real Generator + Discriminator training on Fashion-MNIST |
| ✅ **Visible evidence** | Generated image grid + Diversity score + D-score metric + Loss curves + Progression gallery |
| ✅ **Working submission** | All source files + `requirements.txt` + this README |

---

## Project Structure

```
dcgan_fashion/
├── app.py              ← Streamlit frontend (run this)
├── model.py            ← Generator + Discriminator definitions
├── preprocessing.py    ← Data loading + image preprocessing
├── inference.py        ← Model loading + generation + scoring
├── train.py            ← Training loop + checkpoint saving
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch app
streamlit run app.py
```

Open **http://localhost:8501**

---

## App Sections

| Section | What you see |
|---------|-------------|
| 🏠 **Overview** | Architecture, KPI metrics, class list, latest sample |
| 🏋️ **Train Model** | Live loss updates, epoch/batch sliders, progress bar |
| 🎨 **Generate Images** | Image grid, diversity score, D-score, progression gallery |
| 🔍 **Analyse Uploaded Image** | Upload any clothing photo → Discriminator scores it |
| 📊 **Loss Curves** | G vs D loss chart with interpretation guide |

---

## CLI Training (without UI)

```bash
python train.py --epochs 10 --batch_size 64 --subset_size 10000 --save_every 2
```

---

## Architecture

**Generator**
```
Noise z (100,1,1) → ConvTranspose2d×3 → Tanh → Image (1,28,28)
```

**Discriminator**
```
Image (1,28,28) → Conv2d×3 → Sigmoid → Score [0,1]
```

Loss: BCELoss | Optimiser: Adam (lr=2e-4, β=0.5, 0.999) | Init: N(0, 0.02)

---

## What Good Training Looks Like

- D loss ≈ **0.693** (ln 2) → balanced adversarial game
- G loss **decreasing** → generator improving
- Generated images show progressively clearer clothing shapes after 5–10 epochs
