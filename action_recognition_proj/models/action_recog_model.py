import joblib

model = joblib.load("")

def predict_action(data):
    prediction = model.predict(data)
    return prediction