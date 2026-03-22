import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. โหลดข้อมูล
# ==========================================
df = pd.read_csv('bank.csv')
print(f"จำนวนข้อมูลเริ่มต้น: {len(df)} แถว")

# ==========================================
# 2. EDA & จัดการ Outlier (หมวด 2: 5 คะแนน)
# ==========================================
def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

# ตัด Outlier ของตัวแปร balance และ campaign 
# (เหตุผล: ป้องกันโมเดลเรียนรู้จากลูกค้า VIP ที่เงินเยอะผิดปกติ หรือเคสที่ถูกโทรตื๊อมากเกินไป)
df_clean = remove_outliers_iqr(df, 'balance')
df_clean = remove_outliers_iqr(df_clean, 'campaign')
print(f"จำนวนข้อมูลหลังตัด Outlier: {len(df_clean)} แถว")

# ตัด Data Leakage ทิ้ง
# (เหตุผล: เราจะไม่รู้ระยะเวลาคุยสาย 'duration' ก่อนที่จะโทรหาลูกค้าจริง)
df_clean = df_clean.drop(columns=['duration'])

# ==========================================
# 3. เตรียมข้อมูล Train/Test และ Data Pipeline
# ==========================================
X = df_clean.drop('deposit', axis=1)
y = df_clean['deposit'].map({'yes': 1, 'no': 0}) # แปลง Target เป็น 1, 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# กำหนดกลุ่มตัวแปร (เหลือ 15 ตัวแปร)
numeric_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# สร้าง Pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==========================================
# 4. Model Training & Hyperparameter Tuning (หมวด 3: 5 คะแนน)
# ==========================================
# ใช้ Random Forest และจัดการ Imbalanced Data ด้วย class_weight='balanced'
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# หาพารามิเตอร์ที่ดีที่สุด (ใช้ cv=5 และประเมินด้วย f1-score)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 15, None]
}

print("\nกำลังเทรนและจูนโมเดล (อาจใช้เวลาสักครู่)...")
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"พารามิเตอร์ที่ดีที่สุด: {grid_search.best_params_}")

# ==========================================
# 5. ประเมินผล (Evaluation)
# ==========================================
y_pred = best_model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# แสดง Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Deposit (0)', 'Deposit (1)'])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Confusion Matrix on Test Set')
plt.grid(False)
plt.show()

# ==========================================
# 6. บันทึกโมเดล (Save Model)
# ==========================================
joblib.dump(best_model, 'bank_deposit_model.pkl')
print("\n✅ บันทึกโมเดล 'bank_deposit_model.pkl' เรียบร้อยแล้ว! นำไปใช้ทำ Web App ได้เลย")