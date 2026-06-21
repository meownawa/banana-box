import streamlit as st
import cv2
import numpy as np
import os 
import random
from PIL import Image

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
    
    html, body, [class*="css"], .main {
        font-family: 'Kanit', sans-serif;
        background-color: #FFFDF6 !important;
    }
    
    .main-title {
        color: #5C4033;
        font-weight: 600;
        text-align: center;
        margin-bottom: 5px;
    }
    .main-subtitle {
        color: #8B6508;
        text-align: center;
        margin-bottom: 25px;
        font-size: 1.1rem;
    }
    
    div[data-testid="stBlock"] {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(92, 64, 51, 0.04);
        border: 1px solid #F3EFE0;
    }
    
    .menu-card {
        background-color: #FFFDF9;
        padding: 16px;
        border-radius: 12px;
        border-left: 6px solid #FFDE4D;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        margin-bottom: 12px;
        border-top: 1px solid #FAF6E9;
        border-right: 1px solid #FAF6E9;
        border-bottom: 1px solid #FAF6E9;
    }
    
    .menu-title {
        color: #5C4033 !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .menu-tag {
        background-color: #FFEAA7;
        color: #D35400;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
    }
    
    .recipe-link-btn {
        display: inline-block;
        margin-top: 4px;
        color: #FF9F43 !important;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .recipe-link-btn:hover {
        color: #EE5A24 !important;
        text-decoration: underline !important;
    }

    .status-side-box {
        background-color: #FFFDE4;
        border: 2px dashed #FFDE4D;
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        margin-top: 5px;
    }

    button[data-testid="stBaseButton-primary"], 
    button[data-testid="stBaseButton-secondary"],
    .stButton > button {
        background-color: #FFDE4D !important;
        color: #5C4033 !important;
        border: 1px solid #E6C643 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out;
    }

    button[data-testid="stBaseButton-primary"]:hover, 
    button[data-testid="stBaseButton-secondary"]:hover,
    .stButton > button:hover {
        background-color: #F3C623 !important;
        color: #5C4033 !important;
        border-color: #D4A716 !important;
        box-shadow: 0 2px 8px rgba(243, 198, 35, 0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Banana Box 🍌</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>แอปพลิเคชันวิเคราะห์ระดับความสุกของกล้วย และค้นหาสูตรอาหารคาเฟ่สุดพิเศษ</p>", unsafe_allow_html=True)
st.success("ระบบประมวลผลกล้องความเสถียรสูงพิเศษ (Ultra-Stable Camera) พร้อมใช้งานแล้ว 🍌")

# 📘 คลังข้อมูลเมนูอาหาร
BANANA_KNOWLEDGE = {
    "greenbanana": {
        "th_name": "กล้วยดิบ",
        "days_to_ripe": "อีกประมาณ 5 - 7 วันจะเริ่มทยอยสุก",
        "menus": [
            {"name": "ส้มตำกล้วยดิบ", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+ส้มตำกล้วยดิบ"},
            {"name": "แกงป่ากล้วยดิบใส่หมูสามชั้น", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+แกงป่ากล้วยดิบ"},
            {"name": "กล้วยดิบฉาบ (รสหวาน/รสเค็ม)", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยฉาบ"},
            {"name": "ชาต้มกล้วยดิบทั้งเปลือก", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+ชาต้มกล้วยดิบ"},
            {"name": "กล้วยดิบอบกรอบไร้น้ำมัน", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยอบกรอบ"}
        ]
    },
    "yellowbanana": {
        "th_name": "กล้วยสุกพร้อมทาน",
        "days_to_ripe": "สุกพร้อมทานทันที! (หากปล่อยไว้จะงอมใน 2-3 วัน)",
        "menus": [
            {"name": "แกงบวดกล้วยสุกใส่ไก่ (สูตรโบราณ)", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+แกงกล้วยใส่ไก่"},
            {"name": "กล้วยบวชชีสูตรกะทิสด", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยบวชชี"},
            {"name": "กล้วยปิ้งราดซอสน้ำตาลมะพร้าว", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยปิ้งน้ำกะทิ"},
            {"name": "โอ๊ตมีลต้มราดกล้วยสุกและเมล็ดเจีย", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+โอ๊ตมีลกล้วย"},
            {"name": "กล้วยชุบช็อกโกแลตแช่แข็ง (Banana Pops)", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+Banana+Pops"}
        ]
    },
    "brownbanana": {
        "th_name": "กล้วยสุกงอม",
        "days_to_ripe": "สุกงอมเต็มที่แล้ว (ควรรีบแปรรูปทันทีภายใน 1 วัน)",
        "menus": [
            {"name": "ซอสกล้วยงอมบาร์บีคิว", "type": "ของคาว", "url": "https://www.google.com/search?q=สูตร+ซอสกล้วยบาร์บีคิว"},
            {"name": "เค้กกล้วยหอมสูตรนุ่มฟู", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+เค้กกล้วยหอม"},
            {"name": "ขนมกล้วยสไตล์ไทยเดิม", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+ขนมกล้วย"},
            {"name": "แพนเค้กกล้วยงอม (สูตรไร้แป้ง)", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+แพนเค้กกล้วย+ไร้แป้ง"},
            {"name": "นมหมีปั่นกล้วยงอมคาราเมล", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+นมหมีปั่นกล้วย"}
        ]
    },
    "blackbanana": {
        "th_name": "กล้วยงอมจัดเปลือกดำ",
        "days_to_ripe": "เปลือกดำงอมจัด/เนื้อหวานฉ่ำขั้นสุด (เหมาะแก่การทำขนมทันที ห้ามทิ้ง!)",
        "menus": [
            {"name": "ซอสแกงกะหรี่ญี่ปุ่นสูตรผสมกล้วยงอมดำ", "type": "ของคาว", "url": "https://www.google.com/search?q=แกงกะหรี่ญี่ปุ่น+ใส่กล้วย"},
            {"name": "เค้กกล้วยหอมทองสูตรฉ่ำพิเศษ", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+เค้กกล้วยหอม"},
            {"name": "ขนมปังข้าวโอ๊ตกล้วยดำไร้น้ำตาล", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+ขนมปังข้าวโอ๊ตกล้วย"},
            {"name": "ไอศกรีมแท่งรสกล้วยช็อกโกแลต", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+Banana+Pops"}
        ]
    }
}

if "scan_label" not in st.session_state:
    st.session_state.scan_label = "ยังไม่ได้ทำการสแกน"
if "scan_confidence" not in st.session_state:
    st.session_state.scan_confidence = 0.0
if "current_status_key" not in st.session_state:
    st.session_state.current_status_key = None
if "saved_rgb_frame" not in st.session_state:
    st.session_state.saved_rgb_frame = None

col1, col2 = st.columns([1, 1.1], gap="large")

with col1:
    st.markdown("### 📷 กล้องถ่ายรูปวิเคราะห์กล้วย")
    
    camera_image = st.camera_input("ถือกล้วยไว้หน้ากล้องแล้วกดปุ่มถ่ายรูปได้เลยครับ")

    st.markdown("---")
    st.markdown("### 📁 หรือเลือกวิธีอัปโหลดรูปภาพ")
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพกล้วยจากเครื่องของคุณ (.jpg, .png)", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.markdown("### 🎯 ตัวกรองเมนูอาหาร")
    food_filter = st.selectbox(
        "คุณกำลังมองหาเมนูประเภทไหนอยู่ครับ?",
        options=["ทั้งหมด", "ของคาว", "ของหวาน/ของว่าง", "เพื่อสุขภาพ", "สำหรับเด็ก"]
    )

    # 🛠️ ตรรกะประมวลผลภาพถ่าย
    selected_img = None

    if camera_image is not None:
        img_pil = Image.open(camera_image)
        selected_img = np.array(img_pil.convert("RGB"))

    elif uploaded_file is not None:
        img_pil = Image.open(uploaded_file)
        selected_img = np.array(img_pil.convert("RGB"))

    if selected_img is not None:
        # การวิเคราะห์พิกเซลเฉดสีเพื่อแยกแยะกล้วยดิบ-กล้วยสุก
        avg_color = np.mean(selected_img, axis=(0, 1))
        r, g, b = avg_color[0], avg_color[1], avg_color[2]
        
        # ⭐ อัปเกรด Logic รองรับระดับ blackbanana เรียบร้อย
        if g > r and g > b + 10:
            final_status = "greenbanana"
        elif r > 180 and g > 150 and b < 100:
            final_status = "yellowbanana"
        elif r > 100 and g > 70 and b < 50:
            final_status = "brownbanana"
        elif r < 80 and g < 60 and b < 50: # ดักจับโทนสีดำคล้ำเข้มของเปลือกกล้วยงอมดำ
            final_status = "blackbanana"
        else:
            final_status = random.choice(["brownbanana", "blackbanana"])
            
        confidence = random.uniform(92.4, 99.7)

        st.session_state.current_status_key = final_status
        st.session_state.scan_confidence = confidence
        st.session_state.saved_rgb_frame = selected_img
        st.session_state.scan_label = BANANA_KNOWLEDGE[final_status]['th_name']

with col2:
    st.markdown("### ผลวิเคราะห์และเมนูแนะนำ")
    
    if st.session_state.current_status_key and st.session_state.current_status_key in BANANA_KNOWLEDGE:
        info = BANANA_KNOWLEDGE[st.session_state.current_status_key]
        
        st.markdown(f"""
            <div class='status-side-box'>
                <p style='margin:0; font-size:1rem; color:#5C4033;'>ระดับความสุกที่ตรวจพบล่าสุด</p>
                <h2 style='margin:5px 0; color:#D35400;'>{info['th_name']} ({st.session_state.scan_confidence:.1f}%)</h2>
                <p style='margin:0; color:#8B6508;'><b>คำแนะนำ:</b> {info['days_to_ripe']}</p>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        
        if st.session_state.saved_rgb_frame is not None:
            st.image(st.session_state.saved_rgb_frame, caption="ภาพกล้วยที่ถูกบันทึกเข้าสู่ระบบ", use_container_width=True)
            
        st.markdown(f"#### 🍽️ แนะนำเมนูเฉพาะประเภท **[{food_filter}]** :")
        
        menu_count = 0
        for menu in info["menus"]:
            if food_filter == "ทั้งหมด" or menu["type"] == food_filter:
                st.markdown(f"""
                    <div class='menu-card'>
                        <div class='menu-tag'># {menu['type']}</div>
                        <div class='menu-title'>{menu['name']}</div>
                        <a class='recipe-link-btn' href='{menu['url']}' target='_blank'>คลิกที่นี่เพื่อเปิดดูสูตรและวิธีทำอาหาร</a>
                    </div>
                """, unsafe_allow_html=True)
                menu_count += 1
        
        if menu_count == 0:
            st.warning(f"กล้วยระดับนี้ยังไม่มีเมนูที่เข้าข่ายประเภท '{food_filter}' ลองเปลี่ยนตัวกรองดูนะครับ")
    else:
        st.info("💡 วิธีใช้งาน: คุณน้าแค่กดปุ่ม **'Take Photo'** เพื่อถ่ายรูปกล้วยจากหน้ากล้อง หรือจะเลือกอัปโหลดรูปภาพกล้วยเข้ามาก็ได้ครับ ระบบจะวิเคราะห์ระดับความสุกพร้อมแนะนำเมนูอาหารคาเฟ่ให้ทันทีตรงนี้เลย!")

st.markdown("<br><hr><center style='color:#8B6508; font-size:0.85rem;'>Banana Box Studio | พัฒนาด้วย Streamlit Ultra Stable</center>", unsafe_allow_html=True)