# Image Deblurring using Generative Adversarial Networks (GAN) 🖼️✨

This project implements an **Image Deblurring system using Generative Adversarial Networks (GANs)**.  
The goal is to reconstruct **sharp images from motion-blurred inputs** by training a generator–discriminator architecture using deep learning.

The implementation is done in **Python using TensorFlow/Keras and OpenCV**.

---

## 📁 Project Structure

```bash
.
├── image_deblurring_gan.py      # Main training and inference script
├── generated_images/            # Generated images saved during training
├── README.md                    # Project documentation
```

## 📌 Project Overview

The project consists of:

- Loading motion-blurred and sharp image pairs
- Preprocessing images (resizing and normalization)
- Training a GAN model

  -- Generator learns to produce sharp images

  -- Discriminator distinguishes real vs generated images
  
- Generating and saving deblurred outputs
- Testing the trained generator on new blurry images

## 🧰 Software Requirements

### Programming Language
- Python 3.7+
### Required Libraries
```bash
pip install numpy opencv-python matplotlib tensorflow keras
```

## 📊 Dataset
The dataset used for training is sourced from Kaggle:

🔗 Blur Dataset
https://www.kaggle.com/datasets/kwentar/blur-dataset?resource=download

## 🧠 Model Architecture
### Generator

- Dense + Reshape layers
- Upsampling using UpSampling2D
- Convolutional layers with Batch Normalization
- Outputs a 64×64×3 RGB image

### Discriminator

- Convolutional layers with strided downsampling
- Fully connected classifier
- Binary output (real vs fake)

## 📈 Training Details

- Image size: 64 × 64
- Latent vector size: 100
- Batch size: 32
- Epochs: 1000
- Loss function: Binary Cross-Entropy
- Optimizer: Adam

## ⚠️ Notes & Limitations
- This is a basic GAN, not a conditional GAN (cGAN)
- Training stability may vary
- Performance depends heavily on dataset size and quality
- Not optimized for high-resolution images

## 📜 License
- This project is intended for educational and academic purposes.
- You are free to modify and extend the code.

## 👤 Author
- Shakthi Bala

---

### ✅ Why this README is solid
- Matches your **actual code**
- Clear dataset usage
- Explains GAN logic at the right level
- Recruiter- and academic-friendly
- GitHub-rendering safe

If you want next, I can:
- Convert this to **cGAN-based deblurring**
- Improve the **training loop**
- Add **PSNR / SSIM evaluation**
- Make all your READMEs follow a **single personal template**

Just say 👍
