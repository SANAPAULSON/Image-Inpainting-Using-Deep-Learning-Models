import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img, save_img
import matplotlib.pyplot as plt

tf.keras.utils.get_custom_objects()['mse'] = tf.keras.metrics.MeanSquaredError()

ddpm_model = tf.keras.models.load_model("ddpm_denoising.h5")
gan_model = tf.keras.models.load_model("denoising_gan_generator.h5")
autoencoder_model = tf.keras.models.load_model("denoising_autoencoders.h5")
print("Models loaded")


def load_single_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  
    return np.expand_dims(img_array, axis=0)  

def predict_and_save_ddpm(image_path, save_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    denoised_img = ddpm_model.predict(img_array)[0]
    
    denoised_pil = array_to_img(denoised_img)
    denoised_pil.save(save_path)
    print(f"Denoised image (DDPM) saved at: {save_path}")
    return denoised_pil, denoised_img

def predict_with_gan(image_path):
    img_array = load_single_image(image_path)
    denoised_img = gan_model.predict(img_array)
    return denoised_img[0]

def predict_with_autoencoder(image_path):
    img_array = load_single_image(image_path)
    denoised_img = autoencoder_model.predict(img_array)
    return denoised_img[0]

image_path = "noisy_test/00006.jpg" 

ddpm_output_path = "ddpm_denoised_image.jpg"
ddpm_denoised_image, ddpm_img = predict_and_save_ddpm(image_path, ddpm_output_path)

gan_denoised_image = predict_with_gan(image_path)
gan_output_path = "gan_denoised_image.jpg"
save_img(gan_output_path, gan_denoised_image)
print(f"Denoised image (GAN) saved at: {gan_output_path}")

autoencoder_denoised_image = predict_with_autoencoder(image_path)
autoencoder_output_path = "autoencoder_denoised_image.jpg"
save_img(autoencoder_output_path, autoencoder_denoised_image)
print(f"Denoised image (Autoencoder) saved at: {autoencoder_output_path}")

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

noisy_image = load_img(image_path, target_size=(128, 128))
axes[0].imshow(noisy_image)
axes[0].set_title("Noisy Image")
axes[0].axis('off')

axes[1].imshow(ddpm_denoised_image)
axes[1].set_title("DDPM Denoised Image")
axes[1].axis('off')

axes[2].imshow(gan_denoised_image)
axes[2].set_title("GAN Denoised Image")
axes[2].axis('off')

axes[3].imshow(autoencoder_denoised_image)
axes[3].set_title("Autoencoder Denoised Image")
axes[3].axis('off')

plt.tight_layout()
plt.show()

print(f"All denoised images have been saved.")
