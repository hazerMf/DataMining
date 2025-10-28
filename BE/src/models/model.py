from joblib import load

class Model:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = load(model_path)
        self.scaler = load(scaler_path)

    def predict(self, input_data):
        scaled_data = self.scaler.transform(input_data)
        prediction = self.model.predict(scaled_data)
        return prediction

def load_model():
    model_path = 'models/random_forest.joblib'
    scaler_path = 'models/scaler.pkl'
    return Model(model_path, scaler_path)