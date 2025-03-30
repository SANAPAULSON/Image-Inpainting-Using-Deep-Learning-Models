import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import lpips
import torch

# Load LPIPS model (AlexNet used here, but you can also use "vgg" or "squeeze")
lpips_model = lpips.LPIPS(net='alex')

def load_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

def PSNR(input_image, output_image):
    original = Image.fromarray((input_image * 255).astype(np.uint8))
    decrypted = Image.fromarray((output_image * 255).astype(np.uint8))

    original = original.resize((96, 96))
    decrypted = decrypted.resize((96, 96))

    if original.mode != 'L':
        original = original.convert('L')
    if decrypted.mode != 'L':
        decrypted = decrypted.convert('L')

    original_np = np.array(original)
    decrypted_np = np.array(decrypted)

    mse = np.mean((original_np - decrypted_np) ** 2)
    if mse == 0:
        return np.inf  

    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def ssim(input_image, output_image):
    original = Image.fromarray((input_image * 255).astype(np.uint8))
    decrypted = Image.fromarray((output_image * 255).astype(np.uint8))

    original = original.resize((96, 96))
    decrypted = decrypted.resize((96, 96))

    if original.mode != 'L':
        original = original.convert('L')
    if decrypted.mode != 'L':
        decrypted = decrypted.convert('L')

    original_np = np.array(original)
    decrypted_np = np.array(decrypted)

    return structural_similarity(original_np, decrypted_np, multichannel=False)

def compute_lpips(input_image, output_image):
    input_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    output_tensor = torch.tensor(output_image).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    lpips_value = lpips_model(input_tensor, output_tensor).item()
    return lpips_value

# Load images
image_path = "noisy_test/29621.jpg"  
ddpm_output_path = "ddpm_denoised_image.jpg"
gan_output_path = "gan_denoised_image.jpg"
autoencoder_output_path = "autoencoder_denoised_image.jpg"

noisy_img = load_image(image_path)
ddpm_img = load_image(ddpm_output_path)
gan_img = load_image(gan_output_path)
autoencoder_img = load_image(autoencoder_output_path)

# Compute Metrics
ddpm_psnr = PSNR(noisy_img, ddpm_img)
ddpm_ssim = ssim(noisy_img, ddpm_img)
ddpm_lpips = compute_lpips(noisy_img, ddpm_img)

gan_psnr = PSNR(noisy_img, gan_img)
gan_ssim = ssim(noisy_img, gan_img)
gan_lpips = compute_lpips(noisy_img, gan_img)

autoencoder_psnr = PSNR(noisy_img, autoencoder_img)
autoencoder_ssim = ssim(noisy_img, autoencoder_img)
autoencoder_lpips = compute_lpips(noisy_img, autoencoder_img)

# Store Results
metrics_df = pd.DataFrame({
    "Model": ["DDPM", "GAN", "Autoencoder"],
    "PSNR": [ddpm_psnr, gan_psnr, autoencoder_psnr],
    "SSIM": [ddpm_ssim, gan_ssim, autoencoder_ssim],
    "LPIPS": [ddpm_lpips, gan_lpips, autoencoder_lpips]
})

print(metrics_df)

# Identify Best Model
best_psnr_model = metrics_df.iloc[metrics_df['PSNR'].idxmax()]['Model']
best_ssim_model = metrics_df.iloc[metrics_df['SSIM'].idxmax()]['Model']
best_lpips_model = metrics_df.iloc[metrics_df['LPIPS'].idxmin()]['Model']  # Lower LPIPS is better

print(f"Best PSNR performance: {best_psnr_model}")
print(f"Best SSIM performance: {best_ssim_model}")
print(f"Best LPIPS performance: {best_lpips_model}")

# Plot images
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].imshow(noisy_img)
axes[0].set_title("Noisy Image")
axes[0].axis('off')

axes[1].imshow(ddpm_img)
axes[1].set_title("DDPM Denoised Image")
axes[1].axis('off')

axes[2].imshow(gan_img)
axes[2].set_title("GAN Denoised Image")
axes[2].axis('off')

axes[3].imshow(autoencoder_img)
axes[3].set_title("Autoencoder Denoised Image")
axes[3].axis('off')

plt.tight_layout()
plt.show()
