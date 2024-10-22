import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import re
from datetime import datetime, timedelta
from nltk.corpus import stopwords

# Downloads necessários
nltk.download('stopwords')
nltk.download('punkt')

# Carregar o modelo e objetos de pré-processamento
model = joblib.load('modelo_xgb.pkl')
scaler = joblib.load('scaler.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
selector = joblib.load('feature_selector.pkl')

# Função para calcular dias de recesso
def calcular_dias_recesso(data_inicio, data_fim):
    dias_recesso = 0
    data_atual = data_inicio

    while data_atual <= data_fim:
        mes = data_atual.month
        dia = data_atual.day

        # Verificar se é julho ou janeiro
        if mes == 1 or mes == 7:
            dias_recesso += 1
        # Verificar se é dezembro após o dia 20
        elif mes == 12 and dia >= 20:
            dias_recesso += 1

        data_atual += timedelta(days=1)

    return dias_recesso

# Função para extrair tópicos
def extract_topics(description):
    pattern = r'([^,]+)$'
    matches = re.findall(pattern, description)
    return matches if matches else []

# Função para preprocessar texto
def preprocess_text(text_list):
    processed_text = []
    for text in text_list:
        tokens = nltk.word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stopwords.words('portuguese')]
        processed_text.extend(tokens)
    return ' '.join(processed_text)

st.title("Previsão do Tempo de Julgamento de um Processo")

# Entrada para 'classe'
classes_validas = [
    'APELAÇÃO',
    'RECURSO EM SENTIDO ESTRITO',
    'HABEAS CORPUS',
    'EMBARGOS INFRINGENTES E DE NULIDADE',
    'EMBARGOS DE DECLARAÇÃO',
    'CORREIÇÃO PARCIAL',
    'APELAÇÃO CRIMINAL',
    'MANDADO DE SEGURANÇA'
]
classe = st.selectbox("Classe do Processo", classes_validas)

# Entrada para 'ministro_relator'
ministro_relator = st.text_input("Ministro Relator")

# Entrada para 'data_autuacao'
data_autuacao = st.date_input("Data de Autuação", datetime.today())

# Entrada para 'assunto_descricao'
assunto_descricao_input = st.text_area("Descrição do Assunto", help="Insira uma lista de descrições separadas por vírgula.")

# Botão para prever
if st.button("Prever Tempo de Julgamento"):
    # Criar um dataframe com os dados de entrada
    input_data = pd.DataFrame({
        'classe': [classe],
        'ministro_relator': [ministro_relator],
        'data_autuacao': [pd.to_datetime(data_autuacao)],
        'assunto_descricao': [assunto_descricao_input if assunto_descricao_input else 'N/A']
    })

    # Preencher valores nulos
    input_data[['assunto_descricao']] = input_data[['assunto_descricao']].fillna('N/A')

    # Processar 'assunto_descricao'
    input_data['assunto_descricao_extraido'] = input_data['assunto_descricao'].apply(lambda x: extract_topics(x))

    # Preprocessar o texto
    input_data['assunto_descricao_processado'] = input_data['assunto_descricao_extraido'].apply(preprocess_text)

    # Criar 'texto_combinado'
    input_data['texto_combinado'] = input_data['assunto_descricao_processado']

    # Extrair features de data
    input_data['data_autuacao_dias'] = input_data['data_autuacao'].apply(lambda x: x.timestamp() / 86400)
    input_data['ano_autuacao'] = input_data['data_autuacao'].dt.year
    input_data['mes_autuacao'] = input_data['data_autuacao'].dt.month
    input_data['dia_semana_autuacao'] = input_data['data_autuacao'].dt.dayofweek
    input_data['semana_ano_autuacao'] = input_data['data_autuacao'].dt.isocalendar().week

    # Calcular 'dias_recesso' assumindo que a data de decisão seja hoje
    data_decisao = pd.to_datetime(datetime.today())
    input_data['dias_recesso'] = input_data.apply(lambda row: calcular_dias_recesso(row['data_autuacao'], data_decisao), axis=1)

    # Selecionar features numéricas e escalar
    numerical_features = ['data_autuacao_dias', 'ano_autuacao', 'mes_autuacao', 'dia_semana_autuacao', 'semana_ano_autuacao', 'dias_recesso']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # One-Hot Encoding das variáveis categóricas
    input_data_encoded = pd.get_dummies(input_data, columns=['classe', 'ministro_relator'])

    # Criar interações entre 'classe' e 'ministro_relator'
    input_data_encoded['classe_ministro'] = input_data['classe'] + '_' + input_data['ministro_relator']
    input_data_encoded = pd.get_dummies(input_data_encoded, columns=['classe_ministro'])

    # Processar 'texto_combinado' com TF-IDF
    tfidf_vector = vectorizer.transform(input_data['texto_combinado'])
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names_out())

    # Concatenar os dataframes
    input_data_final = pd.concat([input_data_encoded.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # Alinhar as colunas do input com as do modelo
    X_columns = joblib.load('X_columns.pkl')  # Você deve salvar as colunas do X original durante o treinamento
    missing_cols = set(X_columns) - set(input_data_final.columns)
    for col in missing_cols:
        input_data_final[col] = 0
    input_data_final = input_data_final[X_columns]

    # Selecionar as features escolhidas pelo selector
    input_data_selected = selector.transform(input_data_final)

    # Prever o tempo de julgamento logarítmico
    tempo_julgamento_log_pred = model.predict(input_data_selected)

    # Converter para o tempo de julgamento original
    tempo_julgamento_pred = np.expm1(tempo_julgamento_log_pred)

    # Exibir o resultado
    st.subheader("Resultado da Previsão")
    st.write(f"O tempo de julgamento previsto é de **{tempo_julgamento_pred[0]:.2f} dias**.")
