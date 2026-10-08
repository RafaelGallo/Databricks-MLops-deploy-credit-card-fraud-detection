"""Streamlit app for credit card fraud scoring / app Streamlit para pontuação de fraude em cartão de crédito."""

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# model path comes from env var, default is the repo models folder / caminho do modelo vem de variável de ambiente, padrão é a pasta models do repo
MODELO_PATH = Path(os.getenv("MODELO_PATH", Path(__file__).resolve().parent.parent / "models" / "random_forest.joblib"))

CAT_COLS = ["category", "state", "gender"]
NUM_COLS = ["amt", "city_pop", "age", "distance_km", "hour", "day_of_week", "month"]
COLUNAS = CAT_COLS + NUM_COLS

st.set_page_config(page_title="Detecção de fraude", page_icon="💳", layout="wide")


@st.cache_resource
def carregar_modelo(path: str):
    """Load the full pipeline once / carrega o pipeline completo uma vez."""
    return joblib.load(path)


@st.cache_data
def categorias(_modelo):
    """Known categories from the fitted encoder / categorias conhecidas do encoder treinado."""
    enc = _modelo.named_steps["prep"].named_transformers_["cat"]
    return dict(zip(CAT_COLS, [list(c) for c in enc.categories_]))


def classificar(modelo, df: pd.DataFrame, limiar: float) -> pd.DataFrame:
    """Add probability and decision columns / adiciona colunas de probabilidade e decisão."""
    saida = df.copy()
    saida["probabilidade_fraude"] = modelo.predict_proba(df[COLUNAS])[:, 1]
    saida["fraude"] = saida["probabilidade_fraude"] >= limiar
    return saida


if not MODELO_PATH.exists():
    st.error(f"Modelo não encontrado em {MODELO_PATH}")
    st.stop()

modelo = carregar_modelo(str(MODELO_PATH))
cats = categorias(modelo)

st.title("💳 Detecção de fraude em cartão de crédito")
st.caption("Modelo Random Forest treinado no Databricks (SMOTENC) com o dataset Credit Card Fraud Mega Dataset.")

with st.sidebar:
    st.header("Configuração")
    limiar = st.slider("Limiar de decisão", 0.01, 0.99, 0.50, 0.01,
                       help="Probabilidade mínima para classificar como fraude. Menor = mais recall, mais falsos positivos.")
    st.write(f"Modelo: `{MODELO_PATH.name}`")

aba_unica, aba_lote = st.tabs(["Transação única", "Lote (CSV)"])

with aba_unica:
    with st.form("transacao"):
        c1, c2, c3 = st.columns(3)
        category = c1.selectbox("Categoria", cats["category"])
        state = c2.selectbox("Estado", cats["state"], index=cats["state"].index("CA"))
        gender = c3.selectbox("Gênero", cats["gender"])
        amt = c1.number_input("Valor (amt)", min_value=0.0, value=50.0, step=1.0)
        city_pop = c2.number_input("População da cidade", min_value=0, value=100_000, step=1000)
        age = c3.number_input("Idade", min_value=18.0, max_value=110.0, value=40.0, step=1.0)
        distance_km = c1.number_input("Distância cliente-comerciante (km)", min_value=0.0, value=30.0, step=1.0)
        hour = c2.slider("Hora", 0, 23, 12)
        day_of_week = c3.slider("Dia da semana (1=dom ... 7=sáb)", 1, 7, 3)
        month = c1.slider("Mês", 1, 12, 6)
        enviar = st.form_submit_button("Classificar", type="primary")

    if enviar:
        linha = pd.DataFrame([dict(category=category, state=state, gender=gender, amt=amt, city_pop=city_pop,
                                   age=age, distance_km=distance_km, hour=hour, day_of_week=day_of_week, month=month)])
        res = classificar(modelo, linha, limiar).iloc[0]
        m1, m2 = st.columns(2)
        m1.metric("Probabilidade de fraude", f"{res['probabilidade_fraude']:.2%}")
        if res["fraude"]:
            m2.error("🚨 Possível FRAUDE")
        else:
            m2.success("✅ Transação legítima")
        st.progress(min(float(res["probabilidade_fraude"]), 1.0))

with aba_lote:
    st.write("Envie um CSV com as colunas: " + ", ".join(f"`{c}`" for c in COLUNAS))
    arquivo = st.file_uploader("CSV de transações", type="csv")
    if arquivo is not None:
        dados = pd.read_csv(arquivo)
        faltando = [c for c in COLUNAS if c not in dados.columns]
        if faltando:
            st.error(f"Colunas ausentes: {faltando}")
        else:
            res = classificar(modelo, dados, limiar)
            a, b, c = st.columns(3)
            a.metric("Transações", f"{len(res):,}")
            b.metric("Fraudes sinalizadas", f"{int(res['fraude'].sum()):,}")
            c.metric("Taxa sinalizada", f"{res['fraude'].mean():.2%}")
            st.dataframe(res.sort_values("probabilidade_fraude", ascending=False), use_container_width=True)
            st.download_button("Baixar resultado (CSV)", res.to_csv(index=False).encode("utf-8"),
                               "resultado_fraude.csv", "text/csv")
