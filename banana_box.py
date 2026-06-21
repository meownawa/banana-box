import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf
import av 

# 🎨 1. ตั้งค่าหน้าเว็บเบราว์เซอร์ (ใช้ไอคอนกล้วยมาตรฐานสากล)
st.set_page_config(
    page_title="Banana Box", 
    page_icon="🍌", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 💉 2. ฉีด CSS โทนสีคาเฟ่กล้วยหอม (Banana Café Pastel Theme) และตกแต่งปุ่มสีเหลืองตามที่คุณต้องการ
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
    
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

    /* 💛 ปรับแต่งปุ่มในระบบทั้งหมดให้เป็นสีเหลืองคาเฟ่ตามที่คุณเลือก */
    button[data-testid="stBaseButton-primary"], 
    button[data-testid="stBaseButton-secondary"],
    .stButton > button,
    div[data-testid="stWebRtcButton"] > button {
        background-color: #FFDE4D !important;
        color: #5C4033 !important;
        border: 1px solid #E6C643 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out;
    }

    /* เอฟเฟกต์ตอนนำเมาส์ไปวางบนปุ่ม (Hover) */
    button[data-testid="stBaseButton-primary"]:hover, 
    button[data-testid="stBaseButton-secondary"]:hover,
    .stButton > button:hover,
    div[data-testid="stWebRtcButton"] > button:hover {
        background-color: #F3C623 !important;
        color: #5C4033 !important;
        border-color: #D4A716 !important;
        box-shadow: 0 2px 8px rgba(243, 198, 35, 0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Banana Box 🍌</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>แอปพลิเคชันวิเคราะห์ระดับความสุกของกล้วย และค้นหาสูตรอาหารคาเฟ่สุดพิเศษ</p>", unsafe_allow_html=True)

# 1. โหลดโมเดล Keras และ Labels (🛠️ แก้ไขที่อยู่ไฟล์เรียบร้อยเพื่อรองรับระบบ Cloud)
@st.cache_resource
def load_banana_model():
    model_path = "keras_model.h5"
    label_path = "labels.txt"
    model = tf.keras.models.load_model(model_path)
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

try:
    model, labels = load_banana_model()
    st.success("โหลดโมเดล เรียบร้อยพร้อมใช้งาน")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# 2. คลังข้อมูลเมนูอาหาร
BANANA_KNOWLEDGE = {
    "greenbanana": {
        "th_name": "กล้วยดิบ",
        "days_to_ripe": "อีกประมาณ 5 - 7 วันจะเริ่มทยอยสุก",
        "menus": [
            {"name": "ส้มตำกล้วยดิบ", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+ส้มตำกล้วยดิบ"},
            {"name": "แกงป่ากล้วยดิบใส่หมูสามชั้น", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+แกงป่ากล้วยดิบ"},
            {"name": "แหนมเนือง (กล้วยดิบเคียง)", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+แหนมเนือง"},
            {"name": "กล้วยดิบฉาบ (รสหวาน/รสเค็ม)", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยฉาบ"},
            {"name": "กล้วยดิบเบรกแตก", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยเบรกแตก"},
            {"name": "ทองม้วนแป้งกล้วยดิบ", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+ทองม้วนแป้งกล้วย"},
            {"name": "แป้งกล้วยดิบชงดื่มร้อน (รักษาโรคกระเพาะ)", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=ประโยชน์+แป้งกล้วยดิบ"},
            {"name": "ชาต้มกล้วยดิบทั้งเปลือก", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+ชาต้มกล้วยดิบ"},
            {"name": "สมูทตี้กล้วยดิบผงผสมน้ำผึ้ง", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=สูตร+สมูทตี้กล้วยดิบ"},
            {"name": "กล้วยดิบอบกรอบไร้น้ำมัน (สูตรเฟรนช์ฟรายส์)", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยอบกรอบ+หม้อทอดไร้น้ำมัน"},
            {"name": "ข้าวเกรียบกล้วยดิบ", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+ข้าวเกรียบกล้วย"},
            {"name": "บิสกิตแป้งกล้วยดิบเพื่อสุขภาพ", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=สูตร+บิสกิตกล้วย"}
        ]
    },
    "yellowbanana": {
        "th_name": "กล้วยสุกพร้อมทาน",
        "days_to_ripe": "สุกพร้อมทานทันที! (หากปล่อยไว้จะงอมใน 2-3 วัน)",
        "menus": [
            {"name": "แกงบวดกล้วยสุกใส่ไก่ (สูตรโบราณ)", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+แกงกล้วยใส่ไก่"},
            {"name": "ข้าวผัดกล้วยหอมสุกแกงกะหรี่", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+ข้าวผัดกล้วย"},
            {"name": "พิซซ่าหน้ากล้วยสุกและชีส", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+พิซซ่าหน้ากล้วย"},
            {"name": "กล้วยบวชชีสูตรกะทิสด", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยบวชชี"},
            {"name": "กล้วยปิ้งราดซอสน้ำตาลมะพร้าว", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยปิ้งน้ำกะทิ"},
            {"name": "กล้วยทอด (กล้วยแขก)", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยแขก"},
            {"name": "กล้วยสุกกินสดคู่กับกรีกโยเกิร์ต", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=กล้วย+โยเกิร์ต+ลดน้ำหนัก"},
            {"name": "สลัดผลไม้รวมเน้นกล้วยสุกน้ำสลัดใส", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+สลัดผลไม้กล้วย"},
            {"name": "โอ๊ตมีลต้มราดกล้วยสุกและเมล็ดเจีย", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+โอ๊ตมีลกล้วย"},
            {"name": "แซนด์วิชหน้ากล้วยหอมและเนยถั่ว", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+แซนด์วิชกล้วยหอมเนยถั่ว"},
            {"name": "กล้วยชุบช็อกโกแลตแช่แข็ง (Banana Pops)", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+Banana+Pops"},
            {"name": "โตเกียวไส้กล้วยสุกและคัสตาร์ด", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+ขนมโตเกียวกล้วย"}
        ]
    },
    "brownbanana": {
        "th_name": "กล้วยสุกงอม",
        "days_to_ripe": "สุกงอมเต็มที่แล้ว (ควรรีบแปรรูปทันทีภายใน 1 วัน)",
        "menus": [
            {"name": "ซอสกล้วยงอมบาร์บีคิว (ใช้ความหวานแทนน้ำตาล)", "type": "ของคาว", "url": "https://www.google.com/search?q=สูตร+ซอสกล้วยบาร์บีคิว"},
            {"name": "แกงเผ็ดเป็ดย่างใส่เนื้อกล้วยงอม", "type": "ของคาว", "url": "https://www.google.com/search?q=แกงเผ็ดใส่กล้วย"},
            {"name": "หมูอบซอสกล้วยงอมคาราเมล", "type": "ของคาว", "url": "https://www.google.com/search?q=วิธีทำ+หมูอบซอสกล้วย"},
            {"name": "เค้กกล้วยหอมสูตรนุ่มฟู", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+เค้กกล้วยหอม"},
            {"name": "กล้วยตากอบน้ำผึ้งธรรมชาติ", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยตาก"},
            {"name": "ขนมกล้วยสไตล์ไทยเดิม", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+ขนมกล้วย"},
            {"name": "แพนเค้กกล้วยงอม (สูตรไร้แป้ง ใช้ไข่และกล้วย)", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+แพนเค้กกล้วย+ไร้แป้ง"},
            {"name": "คุ้กกี้ข้าวโอ๊ตกล้วยงอมพลังงานสะอาด", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+คุ้กกี้ข้าวโอ๊ตกล้วย"},
            {"name": "สมูทตี้เวย์โปรตีนปั่นใส่กล้วยงอมค้างคืน", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=สูตร+สมูทตี้กล้วยงอม+เวย์"},
            {"name": "ไอศกรีมกล้วยงอมโฮมเมด (ปั่นแช่แข็ง 100%)", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+ไอศกรีมกล้วยหอม"},
            {"name": "นมหมีปั่นกล้วยงอมคาราเมล", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+นมหมีปั่นกล้วย"},
            {"name": "พุดดิ้งกล้วยงอมเนื้อเด้ง", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+พุดดิ้งกล้วย"}
        ]
    },
    "blackbanana": {
        "th_name": "กล้วยงอมจัดเปลือกดำ",
        "days_to_ripe": "เปลือกดำงอมจัด/เนื้อหวานฉ่ำขั้นสุด (เหมาะแก่การทำขนมทันที ห้ามทิ้ง!)",
        "menus": [
            {"name": "ซอสแกงกะหรี่ญี่ปุ่นสูตรผสมกล้วยงอมดำ", "type": "ของคาว", "url": "https://www.google.com/search?q=แกงกะหรี่ญี่ปุ่น+ใส่กล้วย"},
            {"name": "ซอสหมักเนื้อย่างบาร์บีคิวกล้วยดำ", "type": "ของคาว", "url": "https://www.google.com/search?q=สูตร+ซอสกล้วยบาร์บีคิว"},
            {"name": "ข้าวหมกไก่ซอสกล้วยหวานเครื่องเทศ", "type": "ของคาว", "url": "https://www.google.com/search?q=ข้าวหมกไก่"},
            {"name": "เค้กกล้วยหอมทองสูตรฉ่ำพิเศษ", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+เค้กกล้วยหอม"},
            {"name": "กล้วยตากกวนโบราณรสเข้มข้น", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+กล้วยกวน"},
            {"name": "ขนมกล้วยจี่กระทะเนื้อหนึบ", "type": "ของหวาน/ของว่าง", "url": "https://www.google.com/search?q=วิธีทำ+ขนมกล้วยจี่"},
            {"name": "ขนมปังข้าวโอ๊ตกล้วยดำไร้น้ำตาล", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=วิธีทำ+ขนมปังข้าวโอ๊ตกล้วย"},
            {"name": "สมูทตี้ดีท็อกซ์กล้วยงอมจัดผสมผักเคล", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=สูตร+สมูทตี้กล้วย+ดีท็อกซ์"},
            {"name": "โปรตีนบาร์สูตรเนื้อกล้วยดำอบแห้ง", "type": "เพื่อสุขภาพ", "url": "https://www.google.com/search?q=สูตร+โปรตีนบาร์กล้วย"},
            {"name": "ไอศกรีมแท่งรสกล้วยช็อกโกแลต", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+Banana+Pops"},
            {"name": "ขนมปังหน้ากล้วยบดคาราเมลอบชีส", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=ขนมปังหน้ากล้วย"},
            {"name": "นมนมสดกล้วยงอมจัดปั่น", "type": "สำหรับเด็ก", "url": "https://www.google.com/search?q=วิธีทำ+นมกล้วยปั่น"}
        ]
    }
}

class VideoSnapshotProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.latest_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

if "scan_label" not in st.session_state:
    st.session_state.scan_label = "ยังไม่ได้ทำการสแกน"
if "scan_confidence" not in st.session_state:
    st.session_state.scan_confidence = 0.0
if "current_status_key" not in st.session_state:
    st.session_state.current_status_key = None
if "saved_rgb_frame" not in st.session_state:
    st.session_state.saved_rgb_frame = None

# 📊 4. จัด Layout หน้าเว็บ
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### กล้องสแกนกล้วย")
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    cam_col, status_col = st.columns([1.5, 1], gap="small")
    
    with cam_col:
        ctx = webrtc_streamer(
            key="banana-snapshot-scanner",
            video_processor_factory=VideoSnapshotProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
        
    with status_col:
        st.markdown("<div class='status-side-box'>", unsafe_allow_html=True)
        st.markdown("**ผลวิเคราะห์ล่าสุด**")
        st.markdown(f"กลุ่ม: <span style='color:#D35400; font-weight:600;'>{st.session_state.scan_label}</span>", unsafe_allow_html=True)
        if st.session_state.scan_confidence > 0:
            st.markdown(f"ความมั่นใจ: <h3 style='color:#8B6508; margin:0;'>{st.session_state.scan_confidence:.1f}%</h3>", unsafe_allow_html=True)
        else:
            st.markdown("ความมั่นใจ: <h3 style='color:#95a5a6; margin:0;'>-</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### ตัวกรองเมนูอาหาร")
    food_filter = st.selectbox(
        "คุณกำลังมองหาเมนูประเภทไหนอยู่ครับ?",
        options=["ทั้งหมด", "ของคาว", "ของหวาน/ของว่าง", "เพื่อสุขภาพ", "สำหรับเด็ก"]
    )

    confirm_button = st.button("ยืนยันผลการสแกนและค้นหาเมนู", type="primary", use_container_width=True)

with col2:
    st.markdown("### รายละเอียดและเมนูแนะนำ")
    
    if confirm_button:
        if ctx.video_processor and ctx.video_processor.latest_frame is not None:
            snapshot_img = ctx.video_processor.latest_frame
            
            rgb_frame = cv2.cvtColor(snapshot_img, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, (224, 224))
            input_data = np.expand_dims(resized_frame, axis=0)
            input_data = (np.float32(input_data) / 127.5) - 1

            prediction = model.predict(input_data, verbose=0)
            output_data = prediction[0]
            
            max_index = np.argmax(output_data)
            raw_label = labels[max_index]
            
            final_status = "".join([i for i in raw_label if not i.isdigit()]).strip().lower()
            confidence = output_data[max_index] * 100

            st.session_state.current_status_key = final_status
            st.session_state.scan_confidence = confidence
            st.session_state.saved_rgb_frame = rgb_frame
            
            if final_status in BANANA_KNOWLEDGE:
                st.session_state.scan_label = BANANA_KNOWLEDGE[final_status]['th_name']
            else:
                st.session_state.scan_label = final_status.upper()
                
            st.rerun()
        else:
            st.warning("ไม่สามารถบันทึกภาพได้: กรุณากดปุ่ม START ด้านบนเพื่อเปิดสัญญาณกล้องก่อนครับ")

    if st.session_state.current_status_key and st.session_state.current_status_key in BANANA_KNOWLEDGE:
        info = BANANA_KNOWLEDGE[st.session_state.current_status_key]
        
        if st.session_state.saved_rgb_frame is not None:
            st.image(st.session_state.saved_rgb_frame, caption="ภาพที่ถูกบันทึกเพื่อวิเคราะห์", use_container_width=True)
            
        st.info(f"ระดับความสุกที่ตรวจพบ: **{info['th_name']}** ({st.session_state.scan_confidence:.2f}%)")
        st.metric(label="ระยะเวลา/คำแนะนำความสุก", value=info["days_to_ripe"])
        
        st.markdown(f"#### เมนูแนะนำเฉพาะประเภท **[{food_filter}]** :")
        
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
        st.write("แนะนำ: เปิดกล้อง ถือกล้วยให้ชัดเจน แล้วกดปุ่ม **'ยืนยันผลการสแกน'** ฝั่งซ้าย เพื่อดึงรายการสูตรอาหารขึ้นมาแสดงผลในหน้านี้ครับ")

st.markdown("<br><hr><center style='color:#8B6508; font-size:0.85rem;'>Banana Box Studio | พัฒนาด้วย Streamlit + TensorFlow Keras</center>", unsafe_allow_html=True)