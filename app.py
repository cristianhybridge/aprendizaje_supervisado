from flask import Flask, request, jsonify
import pickle
import pandas as pd

modelo = None # Inicializar variable

with open('pipeline.pkl', 'rb') as file:
    modelo = pickle.load(file)

# ================================ API con Flask
app = Flask(__name__)

@app.route('/predecir', methods=['POST'])
def predict():
    # Obtener los datos de la solicitud
    data = request.get_json()

    # Crear un DataFrame de pandas a partir del JSON
    input_data = pd.DataFrame([data])

    # Hacer la predicción usando el modelo que tiene el pipeline que hará la transformación
    prediccion = modelo.predict(input_data)

    # Devolver la predicción como JSON
    output = {'Survived': int(prediccion[0])}

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)

# ================================================