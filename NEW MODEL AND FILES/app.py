import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load decoder from .h5 file
decoder = load_model("vae_celeb/decoder.h5")

# Fixed latent dimension used during training
LATENT_DIM = 128

# Random face generation function
def generate_random_face(seed=None):
    if seed is not None:
        np.random.seed(seed)
    z = np.random.normal(size=(1, LATENT_DIM))
    generated_img = decoder.predict(z)[0]
    generated_img = np.clip(generated_img * 255, 0, 255).astype(np.uint8)
    return generated_img

# Slider-based face generation function (first 10 dims controlled, rest zeroed)
def generate_from_sliders(**kwargs):
    z_input = np.zeros((1, LATENT_DIM))
    for i in range(10):
        z_input[0, i] = kwargs.get(f"z{i+1}", 0.0)
    generated_img = decoder.predict(z_input)[0]
    generated_img = np.clip(generated_img * 255, 0, 255).astype(np.uint8)
    return generated_img

# UI: Random face generation
random_face_ui = gr.Interface(
    fn=generate_random_face,
    inputs=gr.Slider(0, 1000, step=1, label="Random Seed", value=42),
    outputs="image",
    title="VAE Face Generator — Random Seed",
    description="Generates a random face using a latent vector drawn from a Gaussian distribution."
)

# UI: Manual latent space sliders
latent_sliders = [
    gr.Slider(-3, 3, value=0.0, step=0.1, label=f"z{i+1}") for i in range(10)
]
slider_face_ui = gr.Interface(
    fn=generate_from_sliders,
    inputs=latent_sliders,
    outputs="image",
    title="VAE Face Generator — Latent Sliders",
    description="Control the first 10 dimensions of the latent vector manually to generate a face."
)

# Combine both UIs
gr.TabbedInterface(
    [random_face_ui, slider_face_ui],
    ["Random Face", "Latent Sliders"]
).launch()