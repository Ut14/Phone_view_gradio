import gradio as gr
import torch
import numpy as np
from PIL import Image
import joblib
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import hf_hub_download

# --- Load CLIP Model & Processor from Hugging Face Hub ---
clip_model = CLIPModel.from_pretrained("Ut14/clip-phone-view", subfolder="clip_model")
clip_processor = CLIPProcessor.from_pretrained("Ut14/clip-phone-view", subfolder="clip_processor")

# --- Download SVM model from Hugging Face Hub ---
svm_model_path = hf_hub_download(repo_id="Ut14/clip-phone-view", filename="svm_phone_view_model.joblib")
svm_model = joblib.load(svm_model_path)

# --- Label Mapping ---
label_map = {0: "Front", 1: "Back", 2: "Side"}

# --- Extract Features ---
def extract_clip_embedding(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.squeeze().numpy()

# --- Prediction Function for Gradio ---
def predict_view(image: Image.Image):
    embedding = extract_clip_embedding(image)
    pred = svm_model.predict([embedding])[0]
    return label_map[pred]

# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict_view,
    inputs=gr.Image(type="pil", label="Upload Phone Image"),
    outputs=gr.Label(num_top_classes=1, label="Predicted View"),
    title="📱 Phone View Classifier",
    description="Upload an image of a phone (front, back, or side) and get the predicted view using CLIP + SVM."
)

if __name__ == "__main__":
    iface.launch()
