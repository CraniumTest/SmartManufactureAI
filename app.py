from fastapi import FastAPI, HTTPException
from predictive_maintenance import PredictiveMaintenance, LLMInterface
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Welcome to SmartManufacture AI API"}

@app.post("/predict/")
async def predict_failure(data: dict):
    try:
        df = pd.DataFrame(data, index=[0])
        result = predictive_maintenance.predict_failure(df)
        return {"failure_probability": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interpret/")
async def interpret_query(query: str):
    try:
        response = llm_interface.interpret_query(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
