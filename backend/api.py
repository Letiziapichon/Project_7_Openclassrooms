from starlette.responses import Response
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('..\data\model\lgb.pkl')
clients = pd.read_csv('../data/application_test.csv')
clients_preprocess = pd.read_csv('../data/app_test_no_encoded_data.csv')
train_preprocess = pd.read_csv('../data/sample_app_train_no_encoded_data.csv')
data_model = pd.read_csv('../data/clients_data.csv')

def model_score(data):
    return model.predict_proba(data)[0][0]

@app.get('/')
def get_root():
	return {'message': 'Welcome to the credit score API'}

@app.get('/raw_data/')
async def get_raw_data(selected_id: int):
    data_client = clients[clients.SK_ID_CURR == selected_id]
    return Response(data_client.to_json(orient="records"), media_type="application/json")

@app.get('/scoring/')
async def get_scoring(selected_id: int):
    data = data_model[data_model.SK_ID_CURR == selected_id]
    data.drop(columns=['SK_ID_CURR'], inplace=True)
    score =  model_score(data)
    return score

@app.get('/fi_model/')
async def get_fi_model():
    #data = data_model.drop(columns=['SK_ID_CURR'])
    features_names = data_model.columns
    d = {features_names[idx] :  model.feature_importances_[idx] for idx, val in enumerate(model.feature_importances_)}
    df = pd.DataFrame(d.items(), columns=['feature', 'value'])
    return Response(df.to_json(orient="records"), media_type="application/json")


@app.get('/client_data/')
async def get_client_data(selected_id: int):
    idx = data_model[data_model.SK_ID_CURR == int(selected_id)].index.values[0]
    #data_model.drop(columns=['SK_ID_CURR'], inplace=True)
    data = data_model.drop(columns=['SK_ID_CURR']).iloc[idx]
    return Response(data.to_json(), media_type="application/json")

@app.get('/columns/')
async def get_columns(column_type: str):
    if column_type ==  'num':
        columns = list(clients_preprocess.select_dtypes(['int64', 'float64']).columns)[1:]
    else:
        columns = list(clients_preprocess.select_dtypes(['object']).columns)
    return columns

@app.get('/plot_data/')
async def get_columns(column_1: str, column_2: str=None):
    if column_1 == column_2:
        column_2 = None
    columns = [column_1, "TARGET"]
    if column_2 != None:
        columns.append(column_2)
    data = train_preprocess[columns]
    return Response(data.to_json(orient="records"), media_type="application/json")

@app.get('/client_value/')
async def get_columns(selected_id: int, column:str):
    data = clients_preprocess[clients_preprocess.SK_ID_CURR == selected_id][column]
    return Response(data.to_json(orient="records"), media_type="application/json")