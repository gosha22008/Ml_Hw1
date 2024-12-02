from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from pickle import load
import numpy as np
import pandas as pd


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


#загрузка SimpleImputer из файла .pkl
def load_imputer(file_name):
    with open(file_name, 'rb') as imputer_file:
        imputer = load(imputer_file)
    return imputer

#загрузка OneHotEncoding из файла .pkl
def load_encoder(file_name):
    with open(file_name, 'rb') as encoder_file:
        encoder = load(encoder_file)
    return encoder

#загрузка StandartScaler из файла .pkl
def load_scaler(file_name):
    with open(file_name, 'rb') as scaler_file:
        scaler = load(scaler_file)
    return scaler

#загрузка модели из файла .pkl
def load_model(file_name):
    with open(file_name, 'rb') as model_file:
        model = load(model_file)
    return model

#предобработка данных
def preproccessing_data(data, file_true=False):

    #если передан параметр file_true значит передали file.csv
    if file_true:
        data = pd.read_csv(data.file)

    #преобразование данных в DataFrame
    else: data = pd.DataFrame(data, index=[0])

    #удаление единиц измерения из данных и приведение к типу float
    def drop_km(df, features:list):
        for i in features:
            df[i] = df[i].apply(lambda x: x.split()[0] if (isinstance(x, str) and x.split()[0].replace('.','').isdigit()) else np.nan)
            df[i] = df[i].astype(float)

    drop_km(data, ['mileage', 'engine', 'max_power'])

    #удаление столбцов name, torque
    data.drop(['torque', 'name'], axis=1, inplace=True)

    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    #SimpleImputer
    imputer = load_imputer('simpleimputer.pkl')
    data[numeric_cols] = imputer.transform(data[numeric_cols])

    #удаление дублей
    dupli = data.duplicated()
    data = data[~dupli]

    #сброс индекса
    data= data.reset_index(drop=True)

    #приведение типов
    data['engine'] = data['engine'].astype(int)
    data['seats'] = data['seats'].astype(int)

    #категориальные столбцы
    cat_cols = data.select_dtypes([object]).columns.tolist()
    cat_cols.append('seats')

    #OneHotEncoder
    ohe = load_encoder('onehotencoder.pkl')
    df_data_ohe = ohe.transform(data[cat_cols])

    df_data_ohe = pd.DataFrame(df_data_ohe, columns = ohe.get_feature_names_out(cat_cols))
    df_data_cat = data.drop(cat_cols, axis=1)
    df_data_cat = pd.concat([df_data_cat, df_data_ohe], axis=1)
    
    #удаление столбца selling_price
    X = df_data_cat.drop('selling_price', axis=1)

    #StandardScaler
    scal = load_scaler('standartscaler.pkl')
    X_scal = scal.transform(X)
    return X_scal

# предсказание моделью 
def get_predict_price(data):
    file_true=1
    if 'filename' not in dir(data):
        file_true=0
        data = data.dict()
    data = preproccessing_data(data, file_true)
    model = load_model('model.pkl') 
    prediction = model.predict(data)
    return prediction
    
# предсказать значение одного объекта
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    print(dir(item))
    return get_predict_price(item)

# предсказать объекты из csv файла
@app.post("/predict_items")
def predict_items(file: UploadFile=File(description="Загрузите CSV файл")) -> List[float]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Поддерживается только тип файла csv")
    if file:
        prediction = get_predict_price(file)
    return prediction