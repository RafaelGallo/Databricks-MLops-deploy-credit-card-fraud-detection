"""Daily model health check DAG / DAG diária de checagem de saúde do modelo."""

import json
import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task

MODELO_PATH = os.getenv("MODELO_PATH", "/opt/airflow/models/random_forest.joblib")
RELATORIO_PATH = os.getenv("RELATORIO_PATH", "/opt/airflow/output/saude_modelo.json")

# a fixed legit-looking and a risky-looking transaction / uma transação com perfil legítimo e outra com perfil de risco
AMOSTRA = [
    dict(category="grocery_pos", state="KS", gender="M", amt=45.0, city_pop=370212, age=36.0, distance_km=20.0, hour=14, day_of_week=3, month=6),
    dict(category="shopping_net", state="CA", gender="F", amt=1200.0, city_pop=1500, age=29.0, distance_km=85.0, hour=2, day_of_week=6, month=11),
]


@dag(
    dag_id="monitoramento_modelo_fraude",
    start_date=datetime(2026, 10, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["fraude", "mlops"],
)
def monitoramento_modelo():
    @task
    def checar_modelo() -> dict:
        """Load the model, score the sample and validate the output / carrega o modelo, pontua a amostra e valida a saída."""
        import joblib
        import pandas as pd

        modelo = joblib.load(MODELO_PATH)
        proba = modelo.predict_proba(pd.DataFrame(AMOSTRA))[:, 1]
        # probabilities must be valid numbers in [0, 1] / probabilidades precisam ser números válidos em [0, 1]
        if not all(0.0 <= p <= 1.0 for p in proba):
            raise ValueError(f"invalid probabilities: {proba}")
        return {
            "modelo": os.path.basename(MODELO_PATH),
            "verificado_em": datetime.now().isoformat(timespec="seconds"),
            "tamanho_bytes": os.path.getsize(MODELO_PATH),
            "probabilidades_amostra": [float(p) for p in proba],
            "status": "ok",
        }

    @task
    def gravar_relatorio(resultado: dict):
        """Write the latest health report / grava o último relatório de saúde."""
        os.makedirs(os.path.dirname(RELATORIO_PATH), exist_ok=True)
        with open(RELATORIO_PATH, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2)

    gravar_relatorio(checar_modelo())


monitoramento_modelo()
