# 📱 Phone View Classifier with CLIP + SVM

A lightweight image classifier that detects whether a phone image shows the **front**, **back**, or **side** view — powered by OpenAI's CLIP vision encoder and a custom SVM classifier.

![demo](https://github.com/Ut14/phone-view-classifier/assets/demo-preview.gif) <!-- Optional: Add a GIF or screenshot of your app -->

---

## 🚀 Live Demo

👉 Try the deployed app on **[Hugging Face Spaces](https://huggingface.co/spaces/Ut14/Phone_view_classfier)**

---

## ✨ Features

- 🔍 Uses **CLIP ViT-B/32** for extracting rich image embeddings
- 🎯 Trained **SVM classifier** on augmented image data for 3-class classification: `Front`, `Back`, `Side`
- 📈 Includes the **entire training pipeline** (`augmentation`, `embedding extraction`, `model training`)
- ⚡ Interactive **Gradio interface** for quick testing

---

## 🧠 Model Architecture

| Component        | Description                                       |
|------------------|---------------------------------------------------|
| Encoder          | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) |
| Classifier       | Support Vector Machine (SVM) using scikit-learn   |
| Training Images  | 5–6 images per class + augmentation               |
| Classes          | Front, Back, Side                                 |

---

## 🧪 Training Pipeline

| Script | Purpose |
|--------|---------|
| `augmentation.py` | Augments phone images using transforms |
| `clip_feature_extraction.py` | Converts images into CLIP embeddings |
| `classifier.py` | Trains and saves the SVM classifier |

> The output SVM model (`svm_phone_view_model.joblib`) and CLIP processor/model are uploaded to 🤗 Hugging Face Hub: [`Ut14/clip-phone-view`](https://huggingface.co/Ut14/clip-phone-view)

---

## 🖥️ Inference Interface

### Run Locally:

1. **Clone the repo**:

   ```bash
   git clone https://github.com/Ut14/phone-view-classifier.git
   cd phone-view-classifier
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch app**:

   ```bash
   python app.py
   ```

4. Open your browser: `http://localhost:7860`

---

## 📦 Requirements

```
gradio
torch
transformers
huggingface_hub
joblib
pillow
```

---

## 🗂 Project Structure

```
.
├── app.py                        # Gradio web app
├── requirements.txt              # Dependencies
├── classifier.py                 # Train SVM on CLIP embeddings
├── clip_feature_extraction.py   # Extract CLIP features from images
├── augmentation.py              # Augment raw images
├── README.md                    # Project readme
```

---

## 🤗 Model Hosting

The trained components are hosted on Hugging Face under:

🧠 **Model Repo**: [Ut14/clip-phone-view](https://huggingface.co/Ut14/clip-phone-view)

```
Ut14/clip-phone-view/
├── clip_model/                   # CLIPModel weights
├── clip_processor/              # CLIPProcessor
└── svm_phone_view_model.joblib  # Trained SVM classifier
```

---

## 📬 Contact

Built by **Utkarsh Tripathy**  
📧 ut140203@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/utkarsh-tripathy/) | [GitHub](https://github.com/Ut14)
