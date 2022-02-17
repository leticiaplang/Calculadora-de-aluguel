import pickle as pk  # para carregar os arquivos serializados
import pandas as pd  # fazer feature engineering manual


class LinearRegression:

    def __init__(self):  # self referencia a classe
        self.model_pipe = pk.load(open("./model_pipe.pkl", 'rb'))

    def predict(self, df):
        return self.model_pipe.predict(df)
