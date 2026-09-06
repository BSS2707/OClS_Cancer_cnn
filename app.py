import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import streamlit as st
from fpdf import FPDF

# -----------------------------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Oncology Studio",
    page_icon="🔬",
    layout="wide"
)

CLASS_LABELS = ["Benign / Normal", "Malignant / Suspicious"]

# -----------------------------------------------------------------------------
# 2. MODEL ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_cancer_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.eval()
    return model

model = load_cancer_model()

# -----------------------------------------------------------------------------
# 3. GRAD-CAM UTILITY
# -----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        class_idx = torch.argmax(output, dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        
        gradients = self.gradients.detach().numpy()[0]
        activations = self.activations.detach().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        confidence = torch.softmax(output, dim=1)[0][class_idx].item() * 100
        return cam, class_idx, confidence

grad_cam = GradCAM(model, model.layer4[-1])

# -----------------------------------------------------------------------------
# 4. IN-MEMORY PDF GENERATOR
# -----------------------------------------------------------------------------
def create_pdf_report(orig_img_np, overlay_img_np, prediction_label, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    
    pdf.cell(0, 10, "AI Diagnostic Summary", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 10, "Explainable CNN Oncology Studio", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    orig_buf = io.BytesIO()
    overlay_buf = io.BytesIO()
    Image.fromarray(orig_img_np).save(orig_buf, format="PNG")
    Image.fromarray(overlay_img_np).save(overlay_buf, format="PNG")
    orig_buf.seek(0)
    overlay_buf.seek(0)
    
    pdf.image(orig_buf, x=15, y=35, w=85)
    pdf.image(overlay_buf, x=110, y=35, w=85)
    
    pdf.set_y(130)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, f"Diagnosis: {prediction_label}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Confidence Score: {confidence:.2f}%", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Disclaimer: AI research prototype for demonstration only.")
    
    return bytes(pdf.output())

# -----------------------------------------------------------------------------
# 5. MAIN STREAMLIT UI
# -----------------------------------------------------------------------------
st.title("🔬 AI Oncology Studio")
st.write("Upload a scan file below to run CNN classification and generate visual heatmaps.")

uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(raw_image).unsqueeze(0)
    
    with st.spinner("Analyzing scan..."):
        heatmap, pred_idx, confidence_val = grad_cam.generate(input_tensor)
        
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        orig_resized_np = np.array(raw_image.resize((224, 224)))
        overlay = cv2.addWeighted(orig_resized_np, 0.6, heatmap_color, 0.4, 0)

    # 1. Main UI Image Display First
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Input Scan")
        st.image(raw_image, width=350)
        
    with col2:
        st.subheader("Grad-CAM Visual Heatmap")
        st.image(overlay, width=350)

    # 2. Colorful Results Block Placed AFTER the Images
    st.markdown("---")
    pred_text = CLASS_LABELS[pred_idx]
    
    # Styled Result Header & Dynamic Progress Bar
    if pred_idx == 1:
        st.markdown(
            f"<h2 style='color: #FF4B4B; margin-bottom: 0;'>🚨 Diagnosis Result: {pred_text}</h2>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h2 style='color: #00CC66; margin-bottom: 0;'>✅ Diagnosis Result: {pred_text}</h2>", 
            unsafe_allow_html=True
        )
        
    st.write(f"**Confidence Score: {confidence_val:.2f}%**")
    
    # Vibrant progress bar replacement for the slider
    st.progress(int(confidence_val))

    # 3. PDF Download Button
    st.markdown("---")
    pdf_bytes = create_pdf_report(orig_resized_np, overlay, pred_text, confidence_val)
    st.download_button(
        label="📄 Download Diagnostic PDF Report",
        data=pdf_bytes,
        file_name="cancer_diagnostic_report.pdf",
        mime="application/pdf"
    )

else:
    st.info("Please upload an image to run CNN diagnosis.")