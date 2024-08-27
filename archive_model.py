from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

model = load('archive_model_traning.joblib')
app = FastAPI()

class ArchData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/arch_preview")
def arch_preview(data: ArchData):
    arch_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    arch_preview = model.preview(arch_data)
    return {"Preview do POST": arch_preview[0]}