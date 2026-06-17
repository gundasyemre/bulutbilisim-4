import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import json

print("Bulut üzerinde veri çekiliyor ve analiz ediliyor...")
# Hazır klinik verisetini çekiyoruz (Benign/Malignant sınıflandırması)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Eksik veri doldurma simülasyonu (Imputation)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

print("Karar Ağacı (Decision Tree) modeli eğitiliyor...")
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Modeli ve doğruluğunu kaydediyoruz
accuracy = model.score(X_test, y_test)
joblib.dump(model, 'health_model.pkl')

print(f"Model başarıyla eğitildi! Doğruluk (Accuracy): %{accuracy*100:.2f}")