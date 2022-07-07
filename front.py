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

    ############ NUMERIC PLOT

    def return_plot(data, column, plot_type, client_value):
        if plot_type == 'Histogram':
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            sns.histplot(data=data[data.TARGET == 0][[column, 'TARGET']], x=column, stat="percent")
            sns.histplot(data=data[data.TARGET == 1][[column, 'TARGET']], x=column, stat="percent", color="orange", alpha=.8)
            plt.axvline(float(client_value), color='green', linestyle='--', linewidth=3)
            plt.legend([f"client {selected_id}", "payed", "did not pay"], loc=0, frameon=True)
            return fig

        if plot_type == 'Point plot':
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            sns.pointplot(x='TARGET', y=column, data=data[[column, 'TARGET']])
            plt.axhline(y=client_value, color='green', linestyle='--', linewidth=5)
            plt.legend([f"client {selected_id}"], loc=0, frameon=True)
            return fig
        
        if plot_type == 'Box plot':
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            sns.boxplot(x='TARGET', y=column, data=data[[column, 'TARGET']])
            plt.axhline(y=client_value, color='green', linestyle='--', linewidth=5)
            plt.legend([f"client {selected_id}"], loc=0, frameon=True)
            return fig

    st.header("Visualize data") 
    col1, col2 = st.columns(2)

    plot_types = ['Histogram', 'Point plot', 'Box plot']
    column_1 = col1.selectbox(label = "Choose a numric column", options = numeric_cols)
    column_2 = col2.selectbox(label = "Choose a second numeric column", options = numeric_cols)

    plot_type_1 = col1.selectbox(label = "Choose a plot type", options = plot_types)
    plot_type_2 = col2.selectbox(label = "Choose a second plot type", options = plot_types)

    ### PLOT 1
    res = req.get(f'http://127.0.0.1:8000/plot_data/?column_1={column_1}')
    train = pd.DataFrame(res.json())

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_1}')
    client_value = res.json()[0]

    fig = return_plot(train, column_1, plot_type_1, client_value)
    col1.pyplot(fig)

    ### PLOT 2
    res = req.get(f'http://127.0.0.1:8000/plot_data/?column_1={column_2}')
    train_2 = pd.DataFrame(res.json())

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_2}')
    client_value = res.json()[0]

    fig_2 = return_plot(train_2, column_2, plot_type_2, client_value)
    col2.pyplot(fig_2)

    ############ CATEGORICAL PLOT

    column_1 = col1.selectbox(label = "Choose a categorical column", options = categorical_cols)
    column_2 = col2.selectbox(label = "Choose a second categorical column", options = categorical_cols)

    ### PLOT 1
    res = req.get(f'http://127.0.0.1:8000/plot_data/?column_1={column_1}')
    train = pd.DataFrame(res.json())

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_1}')
    client_value = res.json()[0]

    data_1 = train[column_1].groupby(train['TARGET']).value_counts(normalize=True).rename('percent').reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.title(f"Client: {client_value}")
    sns.barplot(x=column_1, y="percent", hue="TARGET", data=data_1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    col1.pyplot(fig)

    ### PLOT 2
    res = req.get(f'http://127.0.0.1:8000/plot_data/?column_1={column_2}')
    train_2 = pd.DataFrame(res.json())

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_1}')
    client_value = res.json()[0]

    data_2 = train_2[column_2].groupby(train_2['TARGET']).value_counts(normalize=True).rename('percent').reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.title(f"Client: {client_value}")
    sns.barplot(x=column_2, y="percent", hue="TARGET", data=data_2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    col2.pyplot(fig)

    ################# MULTIVARIATE PLOT

    col1.subheader("Multivariate plot") 
    col2.subheader("") 
    column_1 = col1.selectbox(label = "Choose a column", options = sorted(numeric_cols + categorical_cols))
    column_2 = col1.selectbox(label = "Choose a second column", options = sorted(numeric_cols + categorical_cols))

    res = req.get(f'http://127.0.0.1:8000/plot_data/?column_1={column_1}&column_2={column_2}')
    train_multi = pd.DataFrame(res.json())

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_1}')
    client_value_1 = res.json()[0]

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_2}')
    client_value_2 = res.json()[0]

    col1.subheader(f"Client {selected_id}:")
    col1.text(f"{column_1}: {client_value_1}")
    col1.text(f"{column_2}: {client_value_2}")

    if (column_1 in numeric_cols and column_2 in categorical_cols) or (column_2 in numeric_cols and column_1 in categorical_cols):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.barplot(x=column_1, y=column_2, hue="TARGET", data=train_multi)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        col2.pyplot(fig)

    elif column_1 in numeric_cols and column_2 in numeric_cols:
        fig, ax = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)
        #fig.subplots_adjust(hspace=0.3, wspace=3)
        sns.scatterplot(x=column_1, y=column_2, data=train_multi[train_multi.TARGET==0], ax=ax[0]).set(title='Repaid the loan')
        sns.scatterplot(
            x=column_1,
            y=column_2,
            data=train_multi[train_multi.TARGET==1],
            ax=ax[1],
            color='orange').set(title='Not repaid the loan')
        col2.pyplot(fig)

    elif column_1 in categorical_cols and column_2 in categorical_cols:
        train_multi['count'] = 1
        fig, ax = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)
        data = train_multi.groupby(['TARGET', column_1, column_2])['count'].sum().reset_index()
        plt.title(f"Client: {client_value}")
        sns.barplot(x=column_2, y="count", hue=column_1, data=data[data['TARGET'] == 0], ax=ax[0]).set(title='Repaid the loan')
        sns.barplot(x=column_2, y="count", hue=column_1, data=data[data['TARGET'] == 1], ax=ax[1]).set(title='Not repaid the loan')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, fontsize=10)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, fontsize=10)
        col2.pyplot(fig)

