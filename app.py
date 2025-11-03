import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from streamlit_option_menu import option_menu

# ==============================
# 1️⃣ PAGE CONFIG (Không đổi)
# ==============================
st.set_page_config(
    page_title="🍌🍋 Fruit Ripeness Classifier",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 2️⃣ CUSTOM CSS (Không đổi)
# ==============================
st.markdown("""
<style>
body {background-color: #0e1117; color: #fafafa;}
.stApp {background-color: #0e1117;}
h1, h2, h3 {color: #00ffb3;}
.uploaded-image {display: flex; justify-content: center; margin-top: 15px;}
.uploaded-image img {
    width: 220px; 
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0, 255, 179, 0.3);
}
.centered {text-align: center;}
</style>
""", unsafe_allow_html=True)

# ==============================
# 3️⃣ SIDEBAR MENU (Không đổi)
# ==============================
with st.sidebar:
    selected = option_menu(
        menu_title="🍌 Fruit Ripeness AI Dashboard",
        options=["🍏 Predict", "ℹ️ About", "❓ Help"],
        icons=["camera", "info-circle", "question-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#111"},
            "icon": {"color": "#00ffb3", "font-size": "22px"},
            "nav-link": {"font-size": "16px", "color": "white", "margin": "5px", "--hover-color": "#00cc8a"},
            "nav-link-selected": {"background-color": "#00ffb3", "color": "black"},
        }
    )

# ==============================
# 4️⃣ LOAD MODEL (Không đổi)
# ==============================
@st.cache_resource
def load_fruit_model():
    return load_model("efficientnet_banana_mango.h5", compile=False)

model = load_fruit_model()
classes = ['banana_ripe', 'banana_rotten', 'banana_unripe', 'mango_ripe', 'mango_rotten', 'mango_unripe']

# Giữ tên lớp chuẩn hoặc lớp bạn muốn.
LAST_CONV_LAYER_NAME = 'top_activation' 

# ==============================
# 5️⃣ GRAD-CAM FUNCTIONS - CẬP NHẬT FIX CUỐI CÙNG ✅
# ==============================
def get_gradcam(model, img_array, last_conv_layer_name=LAST_CONV_LAYER_NAME):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    try:
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except ValueError as e:
        st.error(f"Lỗi: Không tìm thấy lớp '{last_conv_layer_name}' trong mô hình.")
        return None

    with tf.GradientTape() as tape:
        tape.watch(img_tensor) 
        conv_outputs, predictions = grad_model(img_tensor)
        
        class_index = tf.cast(tf.argmax(predictions[0]), dtype=tf.int32) 
        loss = tf.gather(predictions[0], class_index) 

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)

    # Chuẩn hóa (Normalize)
    max_val = tf.reduce_max(heatmap)
    
    # 🌟 CẢI THIỆN XỬ LÝ HEATMAP YẾU
    if max_val < 1e-5:
        # Nếu heatmap quá yếu (gần như toàn 0), áp dụng kỹ thuật chuẩn hóa mạnh hơn 
        # (Ví dụ: chuẩn hóa bằng tổng hoặc một giá trị epsilon)
        
        heatmap = heatmap / (tf.reduce_sum(heatmap) + 1e-6)
        # Chuẩn hóa lại theo max value của heatmap mới (sau khi chia cho sum)
        max_val_new = tf.reduce_max(heatmap)
        if max_val_new > 0:
             heatmap = heatmap / max_val_new
    elif max_val > 1e-10:
        heatmap /= max_val
    
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(original_img), 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlay), heatmap_resized

# ==============================
# 6️⃣ PAGE: PREDICT - CẬP NHẬT LOGIC CONTOUR ✅
# ==============================
if selected == "🍏 Predict":
    st.title("🍌🍋 Fruit Ripeness Prediction")
    st.write("Upload an image of a banana or mango and let the AI predict its ripeness!")

    uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
        st.image(img, caption="🖼️ Uploaded Image", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        with st.spinner("🔍 AI đang phân tích..."):
            preds = model.predict(img_array)
            preds = np.array(preds)
            class_index = int(np.argmax(preds[0]))
            class_name = classes[class_index]
            confidence = np.max(preds[0])

        st.success(f"✅ **Prediction:** {class_name}")
        st.progress(float(confidence))
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        # --- Top 3 predictions bar chart (Không đổi) ---
        top3_idx = np.argsort(preds[0])[::-1][:3]
        top3_classes = [classes[i] for i in top3_idx]
        top3_scores = [preds[0][i] for i in top3_idx]
        fig = go.Figure(go.Bar(
            x=top3_scores,
            y=top3_classes,
            orientation="h",
            marker_color=["#00ffb3", "#00cc99", "#009977"]
        ))
        fig.update_layout(
            title="📊 Top-3 Predictions",
            xaxis_title="Confidence",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Grad-CAM if rotten ---
        if "rotten" in class_name.lower():
            st.subheader("🔥 Grad-CAM Visualization (Rotten Areas)")
            
            heatmap = get_gradcam(model, img_array)
            
            if heatmap is not None and np.max(heatmap) > 0.0: # Kiểm tra lần cuối Heatmap có dữ liệu không
                overlay_img, heatmap_resized = overlay_gradcam(img, heatmap)
                
                col1, col2 = st.columns(2)

                with col1:
                    st.image(overlay_img, caption="Grad-CAM Overlay (Vùng AI tập trung)", use_container_width=True)

                # --- Contour overlay for highlighting rotten areas ---
                
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                
                # 2. 🌟 GIẢM LÀM MỜ: Chỉ (5, 5) để giữ lại chi tiết hơn
                heatmap_blurry = cv2.GaussianBlur(heatmap_uint8, (5, 5), 0) 
                
                # 3. 🌟 HẠ NGƯỠNG RẤT THẤP (0.2): Để bắt được tín hiệu yếu
                threshold_val = int(255 * 0.2) 
                _, thresh = cv2.threshold(heatmap_blurry, threshold_val, 255, cv2.THRESH_BINARY) 
                
                # 4. Tìm các đường viền
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 5. Vẽ đường viền lên ảnh gốc
                contour_img_cv = np.array(img.copy())
                cv2.drawContours(contour_img_cv, contours, -1, (255, 0, 0), 2) 
                
                contour_final_img = Image.fromarray(contour_img_cv)

                with col2:
                    st.image(contour_final_img, caption="Vùng bị thối (Contour Đỏ)", use_container_width=True)
            else:
                 st.warning("⚠️ Mô hình dự đoán là 'rotten', nhưng Heatmap không có dữ liệu (rất yếu), có thể do đặc điểm quá khác biệt so với dữ liệu huấn luyện.")

        else:
            st.info("💡 Grad-CAM chỉ hiển thị khi dự đoán là **'rotten'** để khoanh vùng khu vực bị thối.")

    else:
        st.info("⬆️ Please upload an image to start prediction.")

# ==============================
# 7️⃣ ABOUT (Không đổi)
# ==============================
elif selected == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **Fruit Ripeness Classifier** — EfficientNetB0 
    
    * Sử dụng mạng nơ-ron **EfficientNetB0** (đã được fine-tune).
    * Hỗ trợ **Chuối** 🍌 và **Xoài** 🥭 (3 giai đoạn mỗi loại: **ripe**, **unripe**, **rotten**).
    * Sử dụng thuật toán **Grad-CAM** để giải thích (eXplainable AI - XAI), làm nổi bật vùng ảnh quan trọng nhất dẫn đến kết quả dự đoán (đặc biệt là vùng bị thối).
    """)

# ==============================
# 8️⃣ HELP (Không đổi)
# ==============================
elif selected == "❓ Help":
    st.title("❓ How to Use")
    st.markdown("""
    1️⃣ Đi đến tab **🍏 Predict** và tải lên một hình ảnh (chuối hoặc xoài).
    
    2️⃣ Xem 3 lớp dự đoán hàng đầu và độ tin cậy.
    
    3️⃣ Nếu lớp dự đoán là **rotten** (bị thối) → **Grad-CAM** sẽ tự động xuất hiện. 
    
    * **Ảnh Overlay (Heatmap):** Vùng màu đỏ/vàng là vùng ảnh có ảnh hưởng **mạnh nhất** đến quyết định "rotten" của AI.
    * **Ảnh Contour (Khoanh vùng):** Sử dụng đường viền màu **Đỏ** để khoanh vùng khu vực "thối" quan trọng nhất theo nhận định của Grad-CAM.
    """)