from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Eğitilen modeli yüklüyoruz
try:
    model = joblib.load('health_model.pkl')
except:
    print("Model bulunamadı. Önce train.py çalıştırılmalı!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gelen tablo verisini al
        data = request.json['features']
        features = np.array(data).reshape(1, -1)
        
        # Tahmin yap
        prediction = model.predict(features)
        result = "İyi Huylu (Benign)" if prediction[0] == 1 else "Zararlı (Pathogenic)"
        
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Sunucuyu 5000 portunda dış dünyaya açıyoruz
    app.run(host='0.0.0.0', port=5000)