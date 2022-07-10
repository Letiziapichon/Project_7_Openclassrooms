import dill
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests as req
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_icon="🔮", page_title="Credit score app")
plt.style.use('classic')
st.title("Credit Score App")

model = joblib.load('data/model/lgb.pkl')

with open('data/explainer', 'rb') as f:
   explainer = dill.load(f)

# text input to process all the rest
selected_id = st.text_input('Client ID', '100001')


@st.cache(suppress_st_warning=True)
def return_raw_data(selected_id):
    res = req.get(f'http://127.0.0.1:8000/raw_data/?selected_id={selected_id}')
    raw_data = res.json()
    return raw_data

@st.cache(suppress_st_warning=True)
def return_score(selected_id):
    res = req.get(f'http://127.0.0.1:8000/scoring/?selected_id={selected_id}')
    score = res.json()
    return score

@st.cache(suppress_st_warning=True)
def return_fi_model():
    res = req.get(f'http://127.0.0.1:8000/fi_model/')
    fi_model = res.json()
    return fi_model

@st.cache(suppress_st_warning=True)
def return_data_model_client(selected_id): 
    res = req.get(f'http://127.0.0.1:8000/client_data/?selected_id={selected_id}')
    data_model_client = res.json()
    return data_model_client

@st.cache(suppress_st_warning=True)
def return_columns(column_type): 
    res = req.get(f'http://127.0.0.1:8000/columns/?column_type={column_type}')
    columns = list(res.json())
    return columns

def features_importance_all_model(dict_values):
    df = pd.DataFrame.from_dict(dict_values)

    df["abs_value"] = df["value"].apply(lambda x: abs(x))
    df["colors"] = df["value"].apply(lambda x: "green" if x > 0 else "red")
    df = df.sort_values("abs_value", ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.barplot(x="value",
                y="feature",
                data=df.head(10),
               palette=df.head(10)["colors"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_title("Feature importance all model", fontsize=10)
    ax.set_ylabel("Feature Name", fontsize=5)
    ax.set_xlabel("Coeff", fontsize=5)
    return fig

@st.cache(suppress_st_warning=True)
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
    col1_info, col2_info = st.columns(2)
    col1_info.header("Client informations")
    col1_info.text(f"Age : {int(raw_data[0]['DAYS_BIRTH'] * - 1 / 365)} yo")
    col1_info.text(f"Gender : {raw_data[0]['CODE_GENDER']}")
    col1_info.text(f"Family status : {raw_data[0]['NAME_FAMILY_STATUS']}")
    col1_info.text(f"Revenu : {raw_data[0]['NAME_INCOME_TYPE']}")
    col1_info.text(f"Occupation : {raw_data[0]['OCCUPATION_TYPE']}")

    col2_info.header("Loan status")
    col2_info.text(f"Montant : {int(raw_data[0]['AMT_CREDIT'])} $")
    col2_info.text(" ")

    with col2_info:
        color = "green" if score > 0.6 else "red" if score < 0.4 else "orange"
        col2_info.markdown(f"<p style='color:green;font-size:30px;margin:0px;padding:0px'>Recommendation : {int(score * 100)}%</p>", unsafe_allow_html=True)
        #st.components.v1.html(f"<p style='color:black;font-size:40px;margin:0px;padding:0px'>Recommendation : <em style='border: 1px solid {color}; background-color:{color}; border-radius:12px; padding-left:10px; padding-right:10px'>{int(score * 100)}%</em></p>")
    
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


    ######################################################################################################################

    st.header("Details of the decision")
    col1_f, col2_f = st.columns(2)
    col1_f.subheader(f"Model most important features")
    col1_f.pyplot(features_importance_all_model(fi_model), use_column_width=True)

    col2_f.subheader(f"Client {selected_id} most important features")

    exp = features_importance_client(data_model_client)
    with col2_f:
        components.html(exp.as_html(), height=500)

    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    #########################################################################################

    ############ NUMERIC PLOT


    def return_plot(data, column, plot_type, client_value):
        if plot_type == 'Histogram':
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            sns.histplot(data=data[data.TARGET == 0][[column, 'TARGET']], x=column, stat="probability")
            sns.histplot(data=data[data.TARGET == 1][[column, 'TARGET']], x=column, stat="probability", color="orange", alpha=.8)
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


    st.header("Visualization of the distribution of customers according to whether they have repaid the loan or not")
    st.markdown(f"<p style='color:#1363DF;font-size:30px;margin:0px;padding:0px'>Numeric features</p>", unsafe_allow_html=True)
    col1_num, col2_num = st.columns(2)

    plot_types = ['Histogram', 'Point plot', 'Box plot']
    column_1 = col1_num.selectbox(label = "Choose a numric column", options = numeric_cols)
    column_2 = col2_num.selectbox(label = "Choose a second numeric column", options = numeric_cols)

    plot_type_1 = col1_num.selectbox(label = "Choose a plot type", options = plot_types)
    plot_type_2 = col2_num.selectbox(label = "Choose a second plot type", options = plot_types)

    ### PLOT 1
    res = req.get(f'http://127.0.0.1:8000/plot_data/?column_1={column_1}')
    train = pd.DataFrame(res.json())

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_1}')
    client_value = res.json()[0]

    fig = return_plot(train, column_1, plot_type_1, client_value)
    col1_num.pyplot(fig)

    ### PLOT 2
    res = req.get(f'http://127.0.0.1:8000/plot_data/?column_1={column_2}')
    train_2 = pd.DataFrame(res.json())

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_2}')
    client_value = res.json()[0]

    fig_2 = return_plot(train_2, column_2, plot_type_2, client_value)
    col2_num.pyplot(fig_2)

    ############ CATEGORICAL PLOT
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.markdown(f"<p style='color:#1363DF;font-size:30px;margin:0px;padding:0px'>Categorical features</p>", unsafe_allow_html=True) 

    col1_cat, col2_cat = st.columns(2)

    column_1 = col1_cat.selectbox(label = "Choose a categorical column", options = categorical_cols)
    column_2 = col2_cat.selectbox(label = "Choose a second categorical column", options = categorical_cols)

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
    col1_cat.pyplot(fig)

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
    col2_cat.pyplot(fig)

    ################# MULTIVARIATE PLOT
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    col1_multi, col2_multi = st.columns(2)
    col1_multi.markdown(f"<p style='color:#1363DF;font-size:30px;margin:0px;padding:0px'>Multifeatures</p>", unsafe_allow_html=True)
    col2_multi.subheader("") 
    column_1 = col1_multi.selectbox(label = "Choose a column", options = sorted(numeric_cols + categorical_cols))
    column_2 = col1_multi.selectbox(label = "Choose a second column", options = sorted(numeric_cols + categorical_cols))

    res = req.get(f'http://127.0.0.1:8000/plot_data/?column_1={column_1}&column_2={column_2}')
    train_multi = pd.DataFrame(res.json())

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_1}')
    client_value_1 = res.json()[0]

    res = req.get(f'http://127.0.0.1:8000/client_value/?selected_id={selected_id}&column={column_2}')
    client_value_2 = res.json()[0]

    col1_multi.subheader(f"Client {selected_id}:")
    col1_multi.text(f"{column_1}: {client_value_1}")
    col1_multi.text(f"{column_2}: {client_value_2}")

    if (column_1 in numeric_cols and column_2 in categorical_cols) or (column_2 in numeric_cols and column_1 in categorical_cols):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.barplot(x=column_1, y=column_2, hue="TARGET", data=train_multi)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        col2_multi.pyplot(fig)

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
        col2_multi.pyplot(fig, figsize=(5,5))

    elif column_1 in categorical_cols and column_2 in categorical_cols:
        train_multi['count'] = 1
        fig, ax = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)
        data = train_multi.groupby(['TARGET', column_1, column_2])['count'].sum().reset_index()
        plt.title(f"Client: {client_value}")
        sns.barplot(x=column_2, y="count", hue=column_1, data=data[data['TARGET'] == 0], ax=ax[0]).set(title='Repaid the loan')
        sns.barplot(x=column_2, y="count", hue=column_1, data=data[data['TARGET'] == 1], ax=ax[1]).set(title='Not repaid the loan')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, fontsize=10)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, fontsize=10)
        col2_multi.pyplot(fig, figsize=(5,5))

