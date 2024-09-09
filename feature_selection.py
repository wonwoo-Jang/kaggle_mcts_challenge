import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터셋 로드 (Breast Cancer 데이터셋 사용)
data = load_breast_cancer()
X, y = data.data, data.target

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 표준화 (SVM은 표준화된 데이터를 선호)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM 모델 학습
svm = SVC(kernel='linear', random_state=42)  # 선형 커널을 사용하여 모델 학습
svm.fit(X_train_scaled, y_train)

# 특성 중요도 평가를 위한 SelectFromModel 사용
selector = SelectFromModel(svm, prefit=True, threshold='mean')  # 평균 이상의 중요도를 가진 특성 선택
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# 선택된 특성 출력
selected_features = selector.get_support(indices=True)
print("선택된 특성 인덱스:", selected_features)
print("선택된 특성 이름:", data.feature_names[selected_features])