import dill
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests as req
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_icon="🔮", page_title="Credit score app")
plt.style.use('fivethirtyeight')
st.title("Credit Score App")

model = joblib.load('data/model/lgb.pkl')
#data_model = pd.read_csv('data/model/clients_data.csv')
#train = pd.read_csv('data/sample_app_trai.csv')

with open('data/explainer', 'rb') as f:
   explainer = dill.load(f)
# Load the datas and caching here to avoid reloading 

# text input to process all the rest
selected_id = st.text_input('Client ID', '100001')


def return_raw_data(selected_id):
    res = req.get(f'http://127.0.0.1:8000/raw_data/?selected_id={selected_id}')
    raw_data = res.json()
    return raw_data


def return_score(selected_id):
    res = req.get(f'http://127.0.0.1:8000/scoring/?selected_id={selected_id}')
    score = res.json()
    return score


def return_fi_model():
    res = req.get(f'http://127.0.0.1:8000/fi_model/')
    fi_model = res.json()
    return fi_model


def return_data_model_client(selected_id): 
    res = req.get(f'http://127.0.0.1:8000/client_data/?selected_id={selected_id}')
    data_model_client = res.json()
    return data_model_client


def return_columns(column_type): 
    res = req.get(f'http://127.0.0.1:8000/columns/?column_type={column_type}')
    columns = list(res.json())
    return columns


def features_importance_all_model(dict_values):
    df = pd.DataFrame.from_dict(dict_values)

    df["abs_value"] = df["value"].apply(lambda x: abs(x))
    df["colors"] = df["value"].apply(lambda x: "green" if x > 0 else "red")
    df = df.sort_values("abs_value", ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.barplot(x="feature",
                y="value",
                data=df.head(10),
               palette=df.head(10)["colors"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_title("Feature importance all model", fontsize=10)
    ax.set_ylabel("Coef", fontsize=5)
    ax.set_xlabel("Feature Name", fontsize=5)
    return fig

def features_importance_client(data_model_client, exp=explainer):
    data_model_client = {el[0]: [el[1]] for el in data_model_client.items()}
    data_model_client = pd.DataFrame.from_dict(data_model_client)
    exp = explainer.explain_instance(
        data_row = data_model_client.iloc[0],
        predict_fn = model.predict_proba, 
        num_features=5
    )
    return exp


raw_data = return_raw_data(selected_id)
if len(raw_data) == 0:
    st.header("The entered customer does not exist")
else:
    score = return_score(selected_id) 
    fi_model = return_fi_model()
    data_model_client = return_data_model_client(selected_id)
    columns_plot = list(pd.DataFrame(fi_model).sort_values('value', ascending=False)['feature'])
    numeric_cols = return_columns('num')
    categorical_cols = return_columns('cat')

    pred = 1 if score > 0.5 else 0

    # Visulazing information
    col1, col2 = st.columns(2)
    col1.header("Client informations")
    col1.text(f"Age : {int(raw_data[0]['DAYS_BIRTH'] * - 1 / 365)} yo")
    col1.text(f"Gender : {raw_data[0]['CODE_GENDER']}")
    col1.text(f"Family status : {raw_data[0]['NAME_FAMILY_STATUS']}")
    col1.text(f"Revenu : {raw_data[0]['NAME_INCOME_TYPE']}")
    col1.text(f"Occupation : {raw_data[0]['OCCUPATION_TYPE']}")

    col2.header("Loan status")
    col2.text(f"Montant : {int(raw_data[0]['AMT_CREDIT'])} $")
    col2.text(" ")
    col2.text(" ")

    with col2:
        color = "green" if score > 0.6 else "red" if score < 0.4 else "orange"
        st.components.v1.html(f"<p style='color:black;font-size:40px;margin:0px;padding:0px'>Recommendation : <em style='border: 1px solid {color}; background-color:{color}; border-radius:12px; padding-left:10px; padding-right:10px'>{int(score * 100)}%</em></p>")


    ######################################################################################################################

    st.header("Details of the decision")
    # st.subheader()
    col1_f, col2_f = st.columns(2)
    col1_f.subheader(f"Model most important features")
    col1_f.pyplot(features_importance_all_model(fi_model))

    col2_f.subheader(f"Client {selected_id} most important features")

    exp = features_importance_client(data_model_client)
    with col2_f:
        components.html(exp.as_html(), height=800)


    #########################################################################################
