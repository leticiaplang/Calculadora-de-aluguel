from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from utils import transform_dict_to_pandas
from model_tree import LinearRegression
from typing import Optional


# instanciando objeto
app = FastAPI()

class Aluguel(BaseModel): # payload
    area: int
    room: int
    bath: int
    garage: int
    property: str
    zone: str

# fazendo as rotas (acoes por meio do @)
@app.get("/") #caminho default com barra
async def calculadora_aluguel():
    return {'message':'calculadora de aluguel'}

#solicitar informações para a api
@app.post("/predict")
async def predict_pipe(aluguel:Aluguel):
    aluguel_dict = aluguel.dict()
    df = transform_dict_to_pandas(aluguel_dict, ['area', 'room', 'bath', 'garage', 'property', 'zone'])
    #carregar regressão linear
    lr = LinearRegression()
    predict_value = lr.predict(df)[0] ##devolve um array
    response_body = {}
    response_body['received_values'] = aluguel_dict
    response_body['predict'] = predict_value
    return response_body
