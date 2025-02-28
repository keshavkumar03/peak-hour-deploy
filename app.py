# from flask import Flask, request, jsonify
# import tensorflow.lite as tflite
# import numpy as np
# app = Flask(__name__)

# # Load TFLite model
# interpreter = tflite.Interpreter(model_path='./peak_hour_model.tflite')
# interpreter.allocate_tensors()
# input_tensor_index = interpreter.get_input_details()[0]['index']
# output_tensor_index = interpreter.get_output_details()[0]['index']
# @app.route('/', methods=['GET'])
# def welcome():
#     return jsonify({"message": "this is a Peak hour detection API"})


# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     input_data = np.array([[data["Voltage"], data["Current"], data["Power"], data["Hour"]]], dtype=np.float32)
    
#     interpreter.set_tensor(input_tensor_index, input_data)
#     interpreter.invoke()
#     prediction = interpreter.get_tensor(output_tensor_index)
    
#     return jsonify({"Peak_Hour": bool(prediction[0] > 0.5)})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime

# Load the TensorFlow SavedModel
model = tf.saved_model.load("saved_model")

# Load the scaler (used during training)
scaler = joblib.load("scaler.pkl")  # Make sure this exists

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features
        timestamp = data.get("timestamp")  # Example: "2025-04-06 12:15:00"
        power_consumption = float(data.get("power_consumption"))

        # Convert timestamp to hour
        hour = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").hour

        # Normalize input
        input_data = np.array([[hour, power_consumption]], dtype=np.float32)
        input_data = scaler.transform(input_data)
        input_data = input_data.reshape(1, 1, 2)  # Reshape for LSTM

        # Run inference
        infer = model.signatures["serving_default"]
        prediction = infer(tf.constant(input_data))["output_0"].numpy()

        # Convert to peak hour decision
        peak_hour = int(prediction[0] > 0.7)

        return jsonify({"timestamp": timestamp, "peak_hour": peak_hour})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
