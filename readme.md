

---

````markdown
# 🧠 Denoising Diffusion GANs and Latent Space Analysis

This project explores **DD-GAN (Denoising Diffusion GANs)**: a generative model combining the power of GANs and diffusion models to generate high-quality and diverse images efficiently. We also perform latent space analysis and semantic attribute editing using a ResNet-18 attribute classifier and latent direction manipulation.

---

## 📌 Abstract

Generative models like VAEs, GANs, and diffusion models have revolutionized AI-driven image synthesis. Yet, each comes with trade-offs in quality, diversity, and efficiency. DD-GAN is a hybrid approach that integrates conditional GANs with diffusion processes to achieve **high-quality, fast-sampled images** while preserving diversity. This project trains DD-GAN on CelebA-HQ, evaluates it using **FID**, and applies **latent space analysis** for semantic editing.

---

## 🔑 Keywords

`DD-GAN`, `Diffusion Models`, `GANs`, `Latent Space`, `Attribute Editing`, `CelebA-HQ`, `ResNet`, `FID`, `Generative AI`

---

## 📐 Architecture

DD-GAN includes the following core components:

### 1. **Forward Diffusion**
- Adds Gaussian noise to real images over **T = 2** timesteps.
- Greatly reduces sampling steps vs traditional 1000-step diffusion models.

### 2. **Conditional Generator (U-Net)**
- Accepts:
  - Noisy image `x_t`
  - Timestep embedding
  - Latent vector `z`
- Predicts denoised image `x_0`

### 3. **Posterior Sampling**
- Uses GAN-learned posterior instead of exact diffusion reverse process
- Allows **fast sampling (2–4 steps)**

### 4. **Discriminator & Training Stability**
- Adversarial loss + R1 regularization
- **EMA (Exponential Moving Average)** for smooth training
- Uses **FID** for evaluation

---

## 🧪 Methodology

### ➤ Dataset
- **CelebA-HQ (128×128)**
- ~5000 images used for training DD-GAN
- Full CelebA dataset used for ResNet attribute classifier (200k images, 40 attributes)

### ➤ Training Setup
| Config           | Value             |
|------------------|------------------|
| GPUs             | 3× NVIDIA A30 (24GB) |
| Epochs           | 1200             |
| Diffusion Steps  | 2                |
| Batch Size       | 32               |
| Learning Rate    | 1e-4 (Dis), 2e-4 (Gen) |

---

## 🛠️ Installation

```bash
git clone https://github.com/your-repo/denoising-diffusion-gan.git
cd denoising-diffusion-gan
conda create -n ddgan python=3.8
conda activate ddgan
pip install -r requirements.txt
````

---

## 📂 Project Structure

```bash
.
├── custom_train_ddgan.py                # DD-GAN training script
├── custom_test.py                # Image generator (from saved model)
├── predict_attributes.py         # Predict attributes using ResNet
├── train_celeba_256_classifier.py  # Train ResNet-18 on CelebA
├── find_latent_directions.py     # Latent direction via logistic regression
├── latent_image_generator.py     # Edits latent vectors based on direction
├── progression.py                # Progressive gain editing
├── attr_verify.py                # Shows images + attributes
├── generated_samples/            # Generated outputs
├── generated_samples_with_latents/
├── datasets/, datasets_prep/     # Data folders
├── pytorch_fid/                  # FID evaluation
├── requirements.txt
```

---

## 🧭 Usage Guide

### 1. ✅ Resize Images

```bash
python image_resize.py
```

### 2. 🚀 Train DD-GAN

```bash
python3 custom_train.py  --dataset custom_128 \--image_size 128 --exp ddgan_custom_exp1   --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --batch_size 32 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema  --r1_gamma 2. --z_emb_dim 256 --lr_d 1e-4  --lr_g 2e-4  --lazy_reg 10  --num_process_per_node 3  --save_content --save_content_every 40 --save_ckpt_every 40 --resume
```

### 3. 🖼️ Generate Samples

```bash
python3 custom_test.py --dataset custom_128 --image_size 128 --exp ddgan_custom_exp1 --num_channels 3 --num_channels_dae 64 \
--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 1200 --batch_size 30000
```

### 4. 🧠 Train Attribute Classifier

```bash
python train_celeba_256_classifier.py --data_root datasets/celeba --attr_file datasets/list_attr_celeba.txt
```

### 5. 📋 Predict Attributes of Generated Samples

```bash
python predict_attributes.py --image_dir generated_samples/
```

### 6. ➡️ Find Latent Directions

```bash
python find_latent_directions.py --attr_file generated_attributes.npy
```

### 7. 🎨 Edit Latent Attributes

Edit `desired_attrs` in `latent_image_generator.py`, then run:

```bash
python latent_image_generator.py
```

### 8. 📈 Visualize Progression

```bash
python progression.py
```

### 9. 🔍 Verify Attributes

```bash
python attr_verify.py
```

---

## 📊 Evaluation: FID Score

To compute FID:

```bash
python -m pytorch_fid generated_samples/ datasets/celeba_128/
```

✅ Achieved FID: **17**

---

## 🧠 Latent Space Analysis

### ✔ Attribute Classifier

* Trained ResNet-18 model to predict 40 attributes
* Achieved attribute prediction loss ≈ **0.01**

### ✔ Logistic Regression for Direction

* Identified latent vector directions using predicted attributes
* Enabled semantic editing (e.g., hair color, age)

### ✔ Editing Results

* Edits like “add blond hair” worked effectively
* **Entanglement** issue observed with sparse/overlapping attributes

---

## 🌍 Applications

* **Creative AI**: Semantic editing of face features
* **Healthcare**: Medical image synthesis
* **Security**: Face reconstruction and enhancement

---

## ⚠️ Challenges

* **Latent Entanglement**: One attribute influences others
* **Limited Data**: Only 5k images used for generator training

---

## 🔮 Future Work

* ✳ Add attribute conditioning during DD-GAN training
* ✳ Use full CelebA-HQ dataset (200k)
* ✳ Explore advanced disentanglement (e.g., InfoGAN, contrastive loss)

---

## 📘 Citation

Xiao, Z., Kreis, K., & Vahdat, A. (2022). *Tackling the Generative Learning Trilemma with Denoising Diffusion GANs*.
**International Conference on Learning Representations (ICLR)**.
🔗 [https://openreview.net/forum?id=LbsDskdxfh](https://openreview.net/forum?id=LbsDskdxfh)

---

## 🧾 License

This codebase is intended for academic and research use only. Based on [official DD-GAN paper](https://openreview.net/forum?id=LbsDskdxfh).

© 2025 Teja Vishnu Vardhan Boddu – Arizona State University

```

---

```
