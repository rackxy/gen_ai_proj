"""
app.py — Frontend: Streamlit interface for Fashion DCGAN
Common Information requirements:
  ✅ 1. Frontend — Streamlit with ≥2 meaningful controls
  ✅ 2. Backend — Separate modules: preprocessing.py, model.py, train.py, inference.py
  ✅ 3. Actual model execution — PyTorch DCGAN (Generator + Discriminator)
  ✅ 4. Visible evidence — Generated images + D-score metric + loss curves + diversity score
  ✅ 5. Working submission — Source + requirements.txt + README

Run: streamlit run app.py
"""

import os, json, glob
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

# ── Backend imports ───────────────────────────────────────────────────────────
from preprocessing import preprocess_uploaded_image, tensor_to_display, FASHION_CLASSES
from inference import (
    load_generator, load_discriminator,
    generate_images, score_image, diversity_score,
    list_generator_checkpoints, list_discriminator_checkpoints,
    get_device,
)
from train import train as run_training

# ── Constants ─────────────────────────────────────────────────────────────────
LOG_FILE   = "loss_log.json"
SAMPLE_DIR = "samples"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion DCGAN Lab",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body, .stApp { background:#0f0f1a; color:#e0e0e0; font-family:'Segoe UI',sans-serif; }
h1,h2,h3 { color:#c084fc; }
.metric-card {
    background:#1e1b4b; border:1px solid #4c1d95; border-radius:12px;
    padding:18px 12px; text-align:center; margin:4px;
}
.metric-val { font-size:1.8rem; font-weight:700; color:#a78bfa; }
.metric-lbl { font-size:0.8rem; color:#9ca3af; margin-top:4px; }
.badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.78rem; font-weight:600; margin:2px;
}
.badge-ok  { background:#14532d; color:#86efac; }
.badge-warn{ background:#78350f; color:#fde68a; }
.stButton>button {
    background:linear-gradient(135deg,#7c3aed,#a855f7);
    color:#fff; border:none; border-radius:8px;
    font-weight:600; padding:0.5rem 1.5rem; width:100%;
}
.stButton>button:hover { opacity:0.85; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar — Control Panel (≥2 meaningful controls) ─────────────────────────
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

# Control 1: Section selector
section = st.sidebar.radio(
    "📂 Section",
    ["🏠 Overview & Architecture",
     "🏋️ Train Model",
     "🎨 Generate Images",
     "🔍 Analyse Uploaded Image",
     "📊 Loss Curves & Metrics"],
)

st.sidebar.markdown("---")

# Control 2: Model selector (checkpoint chooser)
g_ckpts = list_generator_checkpoints()
d_ckpts = list_discriminator_checkpoints()

if g_ckpts:
    selected_g = st.sidebar.selectbox("🤖 Generator Checkpoint", g_ckpts)
    selected_d = st.sidebar.selectbox("🔎 Discriminator Checkpoint", d_ckpts) if d_ckpts else None
    st.sidebar.success(f"✅ {len(g_ckpts)} checkpoint(s) available")
else:
    selected_g = selected_d = None
    st.sidebar.warning("⚠️ No checkpoints — train first")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Device:** `{get_device()}`")
st.sidebar.markdown("**Task 4 · Fashion DCGAN**")


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Overview & Architecture
# ═══════════════════════════════════════════════════════════════════
if section == "🏠 Overview & Architecture":
    st.title("👗 Fashion Image Generator — DCGAN")
    st.markdown(
        "A **Deep Convolutional GAN** trained on **Fashion-MNIST** (10 clothing classes). "
        "Generator and Discriminator compete to produce realistic 28×28 grayscale clothing images."
    )

    # KPI row
    log = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else []
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (len(log), "Epochs Trained"),
        (f"{log[-1]['g_loss']:.4f}" if log else "—", "Last G Loss"),
        (f"{log[-1]['d_loss']:.4f}" if log else "—", "Last D Loss"),
        (len(g_ckpts), "Checkpoints Saved"),
    ]
    for col, (val, lbl) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🧠 Architecture")
        st.markdown("""
**Generator (Generative)**
```
Noise z (100,1,1)
  → ConvTranspose2d → BatchNorm → ReLU  ×3
  → ConvTranspose2d → Tanh
  → Image (1,28,28) in [−1,1]
```

**Discriminator (Discriminative)**
```
Image (1,28,28)
  → Conv2d → BatchNorm → LeakyReLU  ×3
  → Conv2d → Sigmoid
  → Score ∈ [0,1]
```
- **Loss:** Binary Cross Entropy
- **Optimiser:** Adam (lr=2e-4, β=0.5, 0.999)
- **Weight init:** N(0, 0.02) on Conv/BN layers
""")

    with col_b:
        st.subheader("🆚 Generative vs Discriminative")
        st.markdown("""
| | Discriminative | Generative |
|--|--|--|
| **Goal** | Classify real vs fake | Create new samples |
| **Input** | Image | Random noise |
| **Output** | Score / label | New image |
| **Example** | Discriminator (D) | Generator (G) |
| **Training signal** | Direct supervision | Feedback from D |
| **Test case** | Accuracy on test set | Realism of output |

> **Key insight:** D learns the boundary between real and fake;
> G learns to map noise to the data distribution.
""")

    st.markdown("---")
    st.subheader("🗂️ Fashion-MNIST Classes")
    cols = st.columns(5)
    for i, cls in enumerate(FASHION_CLASSES):
        cols[i % 5].markdown(f"`{i}` {cls}")

    samples = sorted(glob.glob(f"{SAMPLE_DIR}/epoch_*.png"))
    if samples:
        st.markdown("---")
        st.subheader("🖼️ Latest Generated Sample Grid")
        st.image(samples[-1], caption=f"Epoch {samples[-1].split('_')[-1].replace('.png','')}", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Train Model
# ═══════════════════════════════════════════════════════════════════
elif section == "🏋️ Train Model":
    st.title("🏋️ Train the DCGAN")
    st.markdown("Configure training parameters and launch. **Checkpoints** and **sample images** are saved automatically.")

    # ── Training controls ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        # Control 3: Epoch slider
        epochs = st.slider("⏱ Epochs", 1, 50, 10, 1,
                            help="More epochs = better quality but longer training")
        # Control 4: Batch size selector
        batch_size = st.selectbox("📦 Batch Size", [32, 64, 128], index=1,
                                   help="Larger batch = more stable but needs more memory")
    with col2:
        # Control 5: Subset slider
        subset_size = st.slider("📊 Training Samples", 1000, 60000, 10000, 1000,
                                 help="Subset of 60k images — smaller runs faster")
        # Control 6: Save frequency
        save_every = st.slider("💾 Save Every N Epochs", 1, 10, 2)

    st.info(
        f"Steps per epoch: **{subset_size // batch_size}** | "
        f"Total steps: **{subset_size // batch_size * epochs}**"
    )

    if st.button("🚀 Start Training"):
        progress_bar = st.progress(0.0, text="Starting...")
        col_g, col_d = st.columns(2)
        g_ph = col_g.empty()
        d_ph = col_d.empty()
        tbl  = st.empty()
        history = []

        def callback(ep, total, g, d):
            progress_bar.progress(ep / total, text=f"Epoch {ep}/{total}")
            g_ph.metric("Generator Loss", f"{g:.4f}",
                        delta=f"{g - history[-1]['g_loss']:.4f}" if history else None)
            d_ph.metric("Discriminator Loss", f"{d:.4f}",
                        delta=f"{d - history[-1]['d_loss']:.4f}" if history else None)
            history.append({"epoch": ep, "g_loss": g, "d_loss": d})
            tbl.dataframe(history[-8:], use_container_width=True)

        run_training(
            epochs=epochs, batch_size=batch_size,
            subset_size=subset_size, save_every=save_every,
            progress_callback=callback,
        )
        st.success("✅ Training complete! Go to **Generate Images** or **Loss Curves**.")
        st.balloons()


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: Generate Images  (Visible evidence: generated output)
# ═══════════════════════════════════════════════════════════════════
elif section == "🎨 Generate Images":
    st.title("🎨 Generate Fashion Images")

    if not selected_g:
        st.warning("No checkpoint loaded. Please **Train** the model first.")
        st.stop()

    G = load_generator(selected_g)
    D = load_discriminator(selected_d) if selected_d else None

    col1, col2 = st.columns([1, 2])
    with col1:
        # Control: number of images
        n_imgs = st.slider("🖼️ Number of images", 4, 64, 16, 4)
        # Control: seed
        seed = st.number_input("🌱 Random seed (0 = random)", 0, 9999, 42)
        # Control: temperature
        temp = st.slider("🌡️ Diversity (temperature)", 0.5, 2.0, 1.0, 0.1,
                         help="Higher = more varied but noisier; lower = more uniform")
        gen_btn = st.button("⚡ Generate!")

    with col2:
        if gen_btn:
            tensors, grid = generate_images(
                G, n=n_imgs,
                seed=int(seed) if seed > 0 else None,
                temperature=temp,
            )

            # ── Visible evidence: generated image grid ────────────────────────
            st.subheader("Generated Fashion Images")
            fig, axes = plt.subplots(
                (n_imgs + 7) // 8, min(8, n_imgs),
                figsize=(min(8, n_imgs) * 1.5, ((n_imgs + 7) // 8) * 1.5)
            )
            fig.patch.set_facecolor("#0f0f1a")
            axes_flat = np.array(axes).flatten()
            for i, ax in enumerate(axes_flat):
                ax.axis("off")
                if i < n_imgs:
                    ax.imshow(tensor_to_display(tensors[i]), cmap="gray")
            plt.tight_layout(pad=0.3)
            st.pyplot(fig)

            # ── Visible evidence: metrics ─────────────────────────────────────
            st.markdown("---")
            st.subheader("📐 Model Output Metrics")
            mc1, mc2, mc3 = st.columns(3)

            div = diversity_score(tensors)
            mc1.metric("Diversity Score", f"{div:.2f}",
                       help="Mean pairwise L2 distance — higher = more varied outputs")

            if D:
                scores = [score_image(D, tensors[i:i+1]) for i in range(min(8, n_imgs))]
                avg_score = np.mean(scores)
                mc2.metric("Avg D-Score (Realism)", f"{avg_score:.3f}",
                           help="Discriminator's average 'real' score for generated images. Closer to 0.5 = well-fooling G")
                label = "🟢 G fooling D" if avg_score > 0.4 else "🟡 D dominating"
                mc3.markdown(f"<div class='metric-card'><div class='metric-val' style='font-size:1rem'>{label}</div><div class='metric-lbl'>Adversarial Balance</div></div>", unsafe_allow_html=True)
            else:
                mc2.info("Train model for D-score")

    # Epoch progression gallery
    st.markdown("---")
    st.subheader("📂 Training Progression")
    sample_imgs = sorted(glob.glob(f"{SAMPLE_DIR}/epoch_*.png"))
    if sample_imgs:
        step = max(1, len(sample_imgs) // 5)
        shown = sample_imgs[::step][-5:]
        cols  = st.columns(len(shown))
        for col, p in zip(cols, shown):
            ep = int(p.split("_")[-1].replace(".png", ""))
            col.image(p, caption=f"Epoch {ep}", use_container_width=True)
    else:
        st.info("Sample images appear here during/after training.")


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Analyse Uploaded Image  (Control: image upload)
# ═══════════════════════════════════════════════════════════════════
elif section == "🔍 Analyse Uploaded Image":
    st.title("🔍 Analyse Your Image")
    st.markdown(
        "Upload any clothing image. The **Discriminator** will score how 'real' it looks "
        "compared to Fashion-MNIST training data."
    )

    # Control: image upload (meaningful control #2)
    uploaded = st.file_uploader(
        "📤 Upload a fashion/clothing image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Try uploading a clothing photo to see how the Discriminator scores it"
    )

    if uploaded and selected_d:
        pil_img = Image.open(uploaded)
        tensor  = preprocess_uploaded_image(pil_img)

        D = load_discriminator(selected_d)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(pil_img, width=200)
            st.caption("Original upload")

            # Show preprocessed version
            arr = tensor_to_display(tensor[0])
            st.image(arr, width=200, caption="After preprocessing (28×28 grayscale, normalised to [−1,1])")

        with col2:
            # ── Visible evidence: D-score on uploaded image ───────────────────
            d_score = score_image(D, tensor)
            st.subheader("Discriminator Analysis")

            st.metric(
                "Realism Score (D output)",
                f"{d_score:.4f}",
                help="1.0 = looks real (like Fashion-MNIST), 0.0 = looks fake"
            )

            # Visual gauge
            fig, ax = plt.subplots(figsize=(5, 1.2))
            fig.patch.set_facecolor("#1e1b4b")
            ax.set_facecolor("#1e1b4b")
            ax.barh(["D-Score"], [d_score], color="#a855f7", height=0.4)
            ax.barh(["D-Score"], [1 - d_score], left=[d_score],
                    color="#2d2860", height=0.4)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="#fbbf24", linewidth=1.5, linestyle="--")
            ax.tick_params(colors="#9ca3af")
            ax.spines[:].set_color("#4c1d95")
            plt.tight_layout()
            st.pyplot(fig)

            if d_score > 0.7:
                verdict = "🟢 Looks very real to the Discriminator"
            elif d_score > 0.4:
                verdict = "🟡 Moderately realistic — D is uncertain"
            else:
                verdict = "🔴 Discriminator thinks this looks fake"
            st.info(verdict)

            st.markdown("""
**Interpretation:**
- Score **> 0.5** → Discriminator believes it's real
- Score **≈ 0.5** → Discriminator is fooled / uncertain
- Score **< 0.5** → Discriminator detects it as fake/OOD
""")

    elif not selected_d:
        st.warning("No Discriminator checkpoint available — train the model first.")
    else:
        st.info("Upload an image above to analyse it.")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: Loss Curves & Metrics  (Visible evidence: diagnostics)
# ═══════════════════════════════════════════════════════════════════
elif section == "📊 Loss Curves & Metrics":
    st.title("📊 Training Loss Curves & Diagnostics")

    if not os.path.exists(LOG_FILE):
        st.warning("No training log found. Please run **Train Model** first.")
        st.stop()

    log = json.load(open(LOG_FILE))
    epochs_list = [r["epoch"]  for r in log]
    g_losses    = [r["g_loss"] for r in log]
    d_losses    = [r["d_loss"] for r in log]

    # ── Visible evidence: loss chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1e1b4b")
    ax.plot(epochs_list, g_losses, "#a855f7", lw=2, marker="o", ms=4, label="Generator Loss")
    ax.plot(epochs_list, d_losses, "#38bdf8", lw=2, marker="s", ms=4, label="Discriminator Loss")
    ax.axhline(0.693, color="#fbbf24", lw=1, linestyle="--", label="Ideal D loss (ln 2 ≈ 0.693)")
    ax.set_xlabel("Epoch", color="#9ca3af")
    ax.set_ylabel("BCE Loss", color="#9ca3af")
    ax.set_title("Generator vs Discriminator Loss", color="#e0e0e0", fontsize=13)
    ax.tick_params(colors="#9ca3af")
    ax.spines[:].set_color("#4c1d95")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(facecolor="#1e1b4b", edgecolor="#4c1d95", labelcolor="#e0e0e0")
    ax.grid(color="#2d2860", ls="--", lw=0.5)
    st.pyplot(fig)

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Epochs",    len(log))
    mc2.metric("Final G Loss",    f"{g_losses[-1]:.4f}")
    mc3.metric("Final D Loss",    f"{d_losses[-1]:.4f}")
    mc4.metric("Best G Loss",     f"{min(g_losses):.4f}")

    st.markdown("---")
    st.subheader("📖 How to Interpret the Curves")
    st.markdown("""
| Pattern | Meaning |
|---------|---------|
| D loss **≈ 0.693** (ln 2) | D is near-random → **ideal adversarial balance** |
| G loss **decreasing** over time | Generator getting better at fooling D |
| D loss **→ 0** | D dominates; G is not learning — try lower D learning rate |
| G loss **→ 0** | G collapsed or D isn't learning — check D architecture |
| Both losses **oscillate** | Normal GAN training behaviour |

**Fashion-MNIST Classes learned:**
T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
""")

    with st.expander("📋 Full Loss Table"):
        st.dataframe(log, use_container_width=True)
