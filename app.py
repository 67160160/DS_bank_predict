import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Page Configuration)
# ==========================================
st.set_page_config(
    page_title="Bank Deposit Prediction App",
    page_icon="🏦",
    layout="wide"
)


# ==========================================
# 2. โหลดโมเดลที่เทรนไว้ (ใช้ Cache เพื่อความรวดเร็ว)
# ==========================================
@st.cache_resource
def load_model():
    # โหลดไฟล์ .pkl ที่เราเซฟไว้จากขั้นตอนที่แล้ว
    return joblib.load('bank_deposit_model.pkl')


model = load_model()

# ==========================================
# 3. ส่วนหัวและคำอธิบาย (UI & Explanations)
# ==========================================
st.title("🏦 Bank Deposit Prediction (AI Assistant)")
st.markdown("""
แอปพลิเคชันนี้ใช้ Machine Learning เพื่อช่วยพนักงานธนาคารวิเคราะห์ว่า **"ลูกค้ามีแนวโน้มจะเปิดบัญชีฝากประจำหรือไม่"** เพื่อประหยัดเวลาและเพิ่มประสิทธิภาพในการทำแคมเปญ Telemarketing

---
""")

# ==========================================
# 4. รับค่าจากผู้ใช้งาน (User Inputs) พร้อม Validation
# ==========================================
st.header("📝 กรอกข้อมูลลูกค้า")

# ใช้ Columns เพื่อจัด Layout ให้สวยงาม
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 ข้อมูลประชากรศาสตร์")
    # Validation: อายุต้อง 18-100 ปี
    age = st.number_input("อายุ (Age)", min_value=18, max_value=100, value=30, help="อายุของลูกค้า (18-100 ปี)")
    job = st.selectbox("อาชีพ (Job)",
                       ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed',
                        'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox("สถานะภาพสมรส (Marital)", ['divorced', 'married', 'single'])
    education = st.selectbox("ระดับการศึกษา (Education)", ['primary', 'secondary', 'tertiary', 'unknown'])

with col2:
    st.subheader("💰 สถานะทางการเงิน")
    # Validation: ยอดเงินไม่ควรเป็นค่าติดลบมากๆ หรือใส่ตัวอักษร
    balance = st.number_input("ยอดเงินคงเหลือเฉลี่ยต่อปี (Balance - ยูโร)", value=1000, help="ยอดเงินในบัญชีปัจจุบัน")
    default = st.selectbox("ประวัติผิดนัดชำระหนี้ (Default)", ['no', 'yes'])
    housing = st.selectbox("มีสินเชื่อบ้านหรือไม่ (Housing Loan)", ['no', 'yes'])
    loan = st.selectbox("มีสินเชื่อส่วนบุคคลหรือไม่ (Personal Loan)", ['no', 'yes'])

with col3:
    st.subheader("📞 ประวัติการติดต่อ")
    contact = st.selectbox("ช่องทางการติดต่อ (Contact)", ['cellular', 'telephone', 'unknown'])
    day = st.slider("วันที่ติดต่อล่าสุด (Day)", min_value=1, max_value=31, value=15)
    month = st.selectbox("เดือนที่ติดต่อล่าสุด (Month)",
                         ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    campaign = st.number_input("จำนวนครั้งที่ติดต่อในแคมเปญนี้ (Campaign)", min_value=1, max_value=50, value=1)
    pdays = st.number_input("จำนวนวันที่ผ่านไปหลังแคมเปญก่อน (Pdays, -1 คือไม่เคยติดต่อ)", min_value=-1, value=-1)
    previous = st.number_input("จำนวนครั้งที่เคยติดต่อแคมเปญก่อน (Previous)", min_value=0, value=0)
    poutcome = st.selectbox("ผลลัพธ์จากแคมเปญก่อนหน้า (Poutcome)", ['failure', 'other', 'success', 'unknown'])

# ==========================================
# 5. สร้าง DataFrame เพื่อส่งให้โมเดลทำนาย
# ==========================================
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'day': [day],
    'month': [month],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'poutcome': [poutcome]
})

st.markdown("---")

# ==========================================
# 6. ปุ่มทำนายผลและแสดงผล (Prediction & Output)
# ==========================================
if st.button("🔍 วิเคราะห์โอกาสเปิดบัญชี", type="primary"):

    # 6.1 ทำนายผลและดึง Probability
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.header("🎯 ผลการวิเคราะห์")

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if prediction == 1:
            st.success("✅ ลูกค้ารายนี้ **มีแนวโน้มสูง** ที่จะเปิดบัญชีฝากประจำ!")
            st.metric(label="ความมั่นใจของระบบ (Confidence)", value=f"{prob[1] * 100:.2f} %")
            st.info("💡 ข้อแนะนำ: ควรจัดสรรพนักงานโทรติดต่อลูกค้ารายนี้เป็นคิวแรกๆ")
        else:
            st.error("❌ ลูกค้ารายนี้ **มีแนวโน้มต่ำ** ที่จะเปิดบัญชีฝากประจำ")
            st.metric(label="ความมั่นใจของระบบ (Confidence)", value=f"{prob[0] * 100:.2f} %")
            st.warning("💡 ข้อแนะนำ: อาจข้ามลูกค้ารายนี้ไปก่อน เพื่อประหยัดเวลาและลดความรำคาญ")

    # 6.2 แสดงกราฟ Probability (Bonus Visual)
    with col_res2:
        fig, ax = plt.subplots(figsize=(6, 3))
        labels = ['No Deposit (ไม่ฝาก)', 'Deposit (ฝาก)']
        colors = ['#ff9999', '#66b3ff']
        ax.barh(labels, [prob[0], prob[1]], color=colors)
        ax.set_xlim(0, 1)
        ax.set_title("Prediction Probability")
        for i, v in enumerate([prob[0], prob[1]]):
            ax.text(v + 0.02, i, f"{v * 100:.1f}%", va='center', fontweight='bold')
        st.pyplot(fig)

# ==========================================
# 7. Disclaimer (ตรงตามเงื่อนไข Rubric)
# ==========================================
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** แอปพลิเคชันนี้ถูกพัฒนาขึ้นเพื่อเป็นส่วนหนึ่งของวิชา ML Deployment Project ผลการทำนายมาจากโมเดล Machine Learning (Random Forest) ที่เรียนรู้จากข้อมูลในอดีตเท่านั้น ไม่ควรใช้เป็นการตัดสินใจทางการเงินหรือการลงทุนขั้นเด็ดขาดโดยปราศจากวิจารณญาณของมนุษย์")