import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. โหลดข้อมูล
df = pd.read_csv('bank.csv')

# --- 📊 ส่วนของ EDA (Exploratory Data Analysis) ---
print("--- 📋 ข้อมูลเบื้องต้น ---")
print(df.info())
print("\n--- 📈 สถิติพื้นฐาน ---")
print(df.describe())

# ตรวจสอบว่า target column ชื่ออะไร (มักเป็น 'deposit' หรือ 'y')
target_col = 'deposit' if 'deposit' in df.columns else 'y'

# สร้างกราฟวิเคราะห์ (จะเซฟเป็นไฟล์รูปภาพ)
plt.figure(figsize=(10, 5))
sns.countplot(x=target_col, data=df, palette='viridis')
plt.title('Distribution of Bank Deposit (Target)')
plt.savefig('target_distribution.png')
print("\n✅ เซฟกราฟการกระจายของเป้าหมายที่ 'target_distribution.png'")

# ดูความสัมพันธ์ของอายุและยอดเงินต่อการเปิดบัญชี
plt.figure(figsize=(12, 6))
sns.scatterplot(x='age', y='balance', hue=target_col, data=df, alpha=0.5)
plt.title('Age vs Balance by Deposit Status')
plt.savefig('age_balance_analysis.png')
print("✅ เซฟกราฟวิเคราะห์ Age vs Balance ที่ 'age_balance_analysis.png'")

# --- ⚙️ การเตรียมข้อมูลสำหรับ Model ---
# กำหนดคอลัมน์ที่จะใช้ทำนาย (เลือกเฉพาะที่หน้าเว็บจะรับค่า)
feature_cols = [
    'age', 'job', 'marital', 'education', 'default', 'balance',
    'housing', 'loan', 'contact', 'day', 'month', 'campaign',
    'pdays', 'previous', 'poutcome'
]

X = df[feature_cols]
y = df[target_col].apply(lambda x: 1 if x == 'yes' else 0) # แปลงเป็น 0, 1

# แบ่งประเภทคอลัมน์
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# 2. สร้างตัวจัดการข้อมูล (Transformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# 3. สร้าง Pipeline (รวมการแปลงข้อมูล + โมเดล)
# การใช้ Pipeline จะแก้ปัญหา KeyError เพราะมันจะจัดการคอลัมน์ให้เราอัตโนมัติ
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=60, random_state=42))
])

# 4. แบ่งข้อมูลเทรนและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. เทรนโมเดล
print("\n⏳ กำลังเทรนโมเดล Pipeline...")
model_pipeline.fit(X_train, y_train)

# 6. ประเมินผล
y_pred = model_pipeline.predict(X_test)
print("\n--- 📊 Performance Report ---")
print(classification_report(y_test, y_pred))

# 7. บันทึกโมเดล Pipeline
joblib.dump(model_pipeline, 'bank_deposit_pipeline.pkl')
print("\n✅ บันทึกโมเดล 'bank_deposit_pipeline.pkl' เรียบร้อย!")