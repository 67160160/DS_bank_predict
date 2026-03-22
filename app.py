import streamlit as st
import pandas as pd
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Bank Deposit Prediction", page_icon="🏦", layout="centered")

st.title("🏦 ระบบทำนายโอกาสเปิดบัญชีเงินฝากประจำ")
st.markdown("กรอกข้อมูลลูกค้าด้านล่าง เพื่อประเมินว่าลูกค้าจะตกลงฝากเงินหรือไม่")

# โหลดโมเดล
@st.cache_resource
def load_model():
    return joblib.load('bank_deposit_model.pkl')

model = load_model()

# สร้างฟอร์มรับข้อมูล
with st.form("prediction_form"):
    st.subheader("👤 ข้อมูลลูกค้า")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("อายุ (Age)", min_value=18, max_value=100, value=30)
        job = st.selectbox("อาชีพ (Job)", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox("สถานะภาพ (Marital)", ['married', 'single', 'divorced'])
        education = st.selectbox("การศึกษา (Education)", ['primary', 'secondary', 'tertiary', 'unknown'])
        default = st.selectbox("มีหนี้เสียหรือไม่ (Default)", ['no', 'yes'])

    with col2:
        balance = st.number_input("ยอดเงินในบัญชี (Balance)", value=1000)
        housing = st.selectbox("มีสินเชื่อบ้านหรือไม่ (Housing)", ['no', 'yes'])
        loan = st.selectbox("มีสินเชื่อส่วนบุคคลหรือไม่ (Loan)", ['no', 'yes'])
        contact = st.selectbox("ช่องทางติดต่อ (Contact)", ['cellular', 'telephone', 'unknown'])
        day = st.number_input("วันที่ติดต่อล่าสุด (Day)", min_value=1, max_value=31, value=15)

    with col3:
        month = st.selectbox("เดือนที่ติดต่อล่าสุด (Month)", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        campaign = st.number_input("จำนวนครั้งที่ติดต่อแคมเปญนี้ (Campaign)", min_value=1, value=1)
        pdays = st.number_input("จำนวนวันหลังติดต่อครั้งก่อน (Pdays)", value=-1)
        previous = st.number_input("จำนวนครั้งที่ติดต่อก่อนหน้านี้ (Previous)", min_value=0, value=0)
        poutcome = st.selectbox("ผลลัพธ์แคมเปญก่อนหน้า (Poutcome)", ['unknown', 'failure', 'other', 'success'])

    submit_button = st.form_submit_button(label="🔮 ทำนายผล")

# เมื่อกดปุ่มทำนาย
if submit_button:
    # 1. จัดเตรียมข้อมูลให้อยู่ในรูปแบบตาราง 2 มิติ
    input_data = pd.DataFrame({
        'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
        'default': [default], 'balance': [balance], 'housing': [housing], 'loan': [loan],
        'contact': [contact], 'day': [day], 'month': [month], 'campaign': [campaign],
        'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome]
    })

    # 2. นำข้อมูลเข้าโมเดล (Pipeline จัดการแปลงข้อมูลให้เองทั้งหมด)
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("📊 ผลการทำนาย")

    # ตรวจสอบผลลัพธ์ว่าฝากหรือไม่ฝาก
    is_deposit = (prediction == 'yes' or prediction == 1)
    
    if is_deposit:
        prob_success = prediction_proba[1] * 100
        st.success(f"**ผลลัพธ์:** ลูกค้ามีแนวโน้ม **'เปิดบัญชีเงินฝาก'** (ความมั่นใจ: {prob_success:.2f}%)")
        st.info("💡 **คำแนะนำธุรกิจ:** ควรติดต่อลูกค้าคนนี้ทันที เสนอโปรโมชั่นพิเศษเพื่อปิดการขายให้ไวที่สุด!")
    else:
        prob_fail = prediction_proba[0] * 100
        st.error(f"**ผลลัพธ์:** ลูกค้ามีแนวโน้ม **'ไม่เปิดบัญชีเงินฝาก'** (ความมั่นใจ: {prob_fail:.2f}%)")
        st.warning("💡 **คำแนะนำธุรกิจ:** ไม่ควรเสียเวลาติดต่อ เสนอให้จัดสรรทรัพยากรไปโฟกัสลูกค้ากลุ่มอื่นที่มีโอกาสมากกว่า")
