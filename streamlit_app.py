import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# 🎨 1. ตั้งค่าหน้าเว็บเบราว์เซอร์
st.set_page_config(
    page_title="Banana Box", 
    page_icon="🍌", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 💉 2. ฉีด CSS โทนสีคาเฟ่กล้วยหอมสุดพรีเมียม
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght=300;400;600&display=swap');
    html, body, [class*="css"], .main { font-family: 'Kanit', sans-serif; background-color: #FFFDF6 !important; }
    .main-title { color: #5C4033; font-weight: 600; text-align: center; margin-bottom: 5px; }
    .main-subtitle { color: #8B6508; text-align: center; margin-bottom: 25px; font-size: 1.1rem; }
    div[data-testid="stBlock"] { background-color: #ffffff; padding: 22px; border-radius: 16px; box-shadow: 0 4px 16px rgba(92, 64, 51, 0.04); border: 1px solid #F3EFE0; }
    .menu-card { background-color: #FFFDF9; padding: 16px; border-radius: 12px; border-left: 6px solid #FFDE4D; box-shadow: 0 2px 8px rgba(0,0,0,0.02); margin-bottom: 12px; }
    .menu-title { color: #5C4033 !important; font-size: 1.1rem; font-weight: 600; margin-bottom: 4px; }
    .menu-tag { background-color: #FFEAA7; color: #D35400; padding: 2px 8px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; margin-bottom: 6px; }
    .recipe-link-btn { display: inline-block; margin-top: 4px; color: #FF9F43 !important; text-decoration: none !important; font-weight: 600; font-size: 0.9rem; }
    .status-side-box { background-color: #FFFDE4; border: 2px dashed #FFDE4D; padding: 12px; border-radius: 12px; text-align: center; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Banana Box 🍌</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>แอปพลิเคชันวิเคราะห์ระดับความสุกของกล้วย ด้วยโมเดล AI ระดับโลกจาก Hugging Face</p>", unsafe_allow_html=True)

# 🧠 3. โหลดโมเดล ViT จาก Hugging Face (บันทึกแคชเพื่อความเร็ว)
@st.cache_resource
def load_hf_model():
    try:
        # ดึงโมเดลจำแนกภาพกล้วยสุกโดยตรงจากคลาวด์ Hugging Face
        classifier = pipeline("image-classification", model="sergiopaniego/vit-banana-ripeness")
        return classifier
    except Exception as e:
        return str(e)

classifier = load_hf_model()

if isinstance(classifier, str):
    st.error(f"เกิดข้อผิดพลาดในการดาวน์โหลดโมเดล: {classifier}")
else:
    st.success("เชื่อมต่อโมเดลระดับสูง Hugging Face (ViT Engine) สำเร็จแล้ว 🍌")

# 📘 คลังข้อมูลเมนูอาหาร (แมตช์ตาม Label ของโมเดลตัวนี้: green, ripe, overripe)
BANANA_KNOWLEDGE = {
    "green": {
        "th_name": "กล้วยดิบ (Green)",
        "days_to_ripe": "อีกประมาณ 5 - 7 วันจะเริ่มทยอยสุก",
        "menus": [
            {"name": "ส้มตำกล้วยดิบ", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+ส้มตำกล้วยดิบ"},
            {"name": "กล้วยดิบฉาบ (รสหวาน/รสเค็ม)", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยฉาบ"}
        ]
    },
    "ripe": {
        "th_name": "กล้วยสุกพร้อมทาน (Ripe)",
        "days_to_ripe": "สุกหวานกำลังดี พร้อมทานทันที!",
        "menus": [
            {"name": "กล้วยบวชชีสูตรกะทิสด", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยบวชชี"},
            {"name": "กล้วยปิ้งราดซอสน้ำตาลมะพร้าว", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยปิ้งน้ำกะทิ"}
        ]
    },
    "overripe": {
        "th_name": "กล้วยสุกงอม/เปลือกดำ (Overripe)",
        "days_to_ripe": "เนื้อหวานฉ่ำขั้นสุด ควรรีบแปรรูปทำขนมทันที ห้ามทิ้ง!",
        "menus": [
            {"name": "เค้กกล้วยหอมสูตรนุ่มฟู", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+เค้กกล้วยหอม"},
            {"name": "แพนเค้กกล้วยงอม (สูตรไร้แป้ง)", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+แพนเค้กกล้วย+ไร้แป้ง"}
        ]
    }
}

if "current_status_key" not in st.session_state: st.session_state.current_status_key = None
if "scan_confidence" not in st.session_state: st.session_state.scan_confidence = 0.0
if "saved_rgb_frame" not in st.session_state: st.session_state.saved_rgb_frame = None

col1, col2 = st.columns([1, 1.1], gap="large")

with col1:
    st.markdown("### 📷 กล้องถ่ายรูปวิเคราะห์กล้วย")
    camera_image = st.camera_input("ถือกล้วยไว้หน้ากล้องแล้วกดถ่ายรูป")
    uploaded_file = st.file_uploader("หรือเลือกไฟล์รูปภาพกล้วยจากเครื่องของคุณ", type=["jpg", "jpeg", "png"])
    food_filter = st.selectbox("คุณกำลังมองหาเมนูประเภทไหนอยู่ครับ?", options=["ทั้งหมด", "ของคาว", "ของหวาน/ของว่าง", "เพื่อสุขภาพ"])

    img_pil = None
    if camera_image is not None:
        img_pil = Image.open(camera_image).convert("RGB")
    elif uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")

    if img_pil is not None and not isinstance(classifier, str):
        # 🛠️ ส่งภาพให้โมเดลของ Hugging Face ทำนายผล
        predictions = classifier(img_pil)
        best_prediction = predictions[0] # ดึงคลาสที่ได้คะแนนสูงสุด
        
        st.session_state.current_status_key = best_prediction['label'] # จะได้ค่า 'green', 'ripe' หรือ 'overripe'
        st.session_state.scan_confidence = best_prediction['score'] * 100
        st.session_state.saved_rgb_frame = np.array(img_pil)

with col2:
    st.markdown("### ผลวิเคราะห์และเมนูแนะนำ")
    if st.session_state.current_status_key in BANANA_KNOWLEDGE:
        info = BANANA_KNOWLEDGE[st.session_state.current_status_key]
        st.markdown(f"""
            <div class='status-side-box'>
                <p style='margin:0; font-size:1rem; color:#5C4033;'>ผลลัพธ์จากโมเดล Hugging Face</p>
                <h2 style='margin:5px 0; color:#D35400;'>{info['th_name']} ({st.session_state.scan_confidence:.1f}%)</h2>
                <p style='margin:0; color:#8B6508;'><b>คำแนะนำ:</b> {info['days_to_ripe']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.saved_rgb_frame is not None:
            st.image(st.session_state.saved_rgb_frame, use_column_width="always")
            
        st.markdown(f"#### 🍽️ แนะนำเมนูเฉพาะประเภท **[{food_filter}]** :")
        for menu in info["menus"]:
            if food_filter == "ทั้งหมด" or menu["type"] == food_filter:
                st.markdown(f"""
                    <div class='menu-card'>
                        <div class='menu-tag'># {menu['type']}</div>
                        <div class='menu-title'>{menu['name']}</div>
                        <a class='recipe-link-btn' href='{menu['url']}' target='_blank'>คลิกเพื่อดูวิธีทำ</a>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("💡 ถ่ายรูปหรืออัปโหลดรูปภาพกล้วยเข้ามาได้เลยครับ ระบบจะส่งรูปภาพไปให้ AI ของ Hugging Face คำนวณผลให้ทันที!")
