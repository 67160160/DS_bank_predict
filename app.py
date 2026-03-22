import streamlit as st
import pandas as pd
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Bank ML Predictor", page_icon="🏦")

st.title("🏦 ระบบทำนายการเปิดบัญชีเงินฝาก")

# 1. โหลดโมเดล (ใช้ Cache เพื่อความเร็ว)
@st.cache_resource
def load_my_model():
    return joblib.load('bank_deposit_pipeline.pkl')

pipeline = load_my_model()

# 2. รับข้อมูลจาก User
st.sidebar.header("📝 กรอกข้อมูลลูกค้า")

def user_input_features():
    age = st.sidebar.slider('อายุ', 18, 95, 35)
    job = st.sidebar.selectbox('อาชีพ', ['admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'])
    marital = st.sidebar.selectbox('สถานะ', ['married','single','divorced'])
    education = st.sidebar.selectbox('การศึกษา', ['primary','secondary','tertiary','unknown'])
    default = st.sidebar.selectbox('มีประวัติค้างชำระ?', ['no','yes'])
    balance = st.sidebar.number_input('ยอดเงินคงเหลือ', value=1000)
    housing = st.sidebar.selectbox('มีเงินกู้บ้าน?', ['no','yes'])
    loan = st.sidebar.selectbox('มีเงินกู้ส่วนบุคคล?', ['no','yes'])
    contact = st.sidebar.selectbox('ช่องทางการติดต่อ', ['cellular','telephone','unknown'])
    day = st.sidebar.slider('วันที่ติดต่อล่าสุด', 1, 31, 15)
    month = st.sidebar.selectbox('เดือนที่ติดต่อล่าสุด', ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    campaign = st.sidebar.number_input('จำนวนครั้งที่ติดต่อในแคมเปญนี้', min_value=1, value=1)
    pdays = st.sidebar.number_input('จำนวนวันที่เว้นช่วงจากแคมเปญก่อน (-1 คือไม่เคยติดต่อ)', value=-1)
    previous = st.sidebar.number_input('จำนวนครั้งที่ติดต่อก่อนหน้านี้', value=0)
    poutcome = st.sidebar.selectbox('ผลลัพธ์จากแคมเปญก่อน', ['unknown','failure','other','success'])

    data = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 3. แสดงข้อมูลที่กรอกและทำนาย
st.subheader("📋 ข้อมูลที่นำมาวิเคราะห์")
st.write(input_df)

if st.button("🔮 ทำนายผล"):
    # ทำนายโดยใช้ Pipeline (มันจะแปลง categorical ให้เราเอง)
    prediction = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df)

    st.markdown("---")
    if prediction[0] == 1:
        st.success(f"🎯 ผลลัพธ์: **มีโอกาสเปิดบัญชีสูง** (ความมั่นใจ {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.error(f"❌ ผลลัพธ์: **มีโอกาสไม่เปิดบัญชี** (ความมั่นใจ {prediction_proba[0][0]*100:.2f}%)")