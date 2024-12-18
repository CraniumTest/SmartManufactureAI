import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PredictiveMaintenance:
    def __init__(self, data_frame):
        self.df = data_frame
        self.model = None

    def preprocess_data(self):
        self.df = self.df.dropna()
        self.features = self.df.drop('failure', axis=1)
        self.labels = self.df['failure']

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        print(classification_report(y_test, predictions))
    
    def predict_failure(self, machine_data):
        return self.model.predict(machine_data)

class LLMInterface:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def interpret_query(self, query):
        inputs = self.tokenizer(query, return_tensors='pt')
        outputs = self.model.generate(inputs.input_ids, max_length=50)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

data_frame = pd.read_csv('machine_data.csv')
predictive_maintenance = PredictiveMaintenance(data_frame)
predictive_maintenance.preprocess_data()
predictive_maintenance.train_model()

llm_interface = LLMInterface()
query = "Predict the failure probability of machine X"
response = llm_interface.interpret_query(query)
print(f"LLM Response: {response}")
