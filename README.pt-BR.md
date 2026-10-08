# MLOps Deploy – Credit Card Fraud Detection

**Português:** [README.pt-BR.md](README.pt-BR.md) · **English:** [README.md](README.md)

Deploy de ponta a ponta de um modelo de detecção de fraude em cartão de crédito: treino no **Databricks** (Delta Lake + MLflow), serviço de predição com **FastAPI**, interface web com **Streamlit**, empacotamento com **Docker** e agendamento com **Apache Airflow**.

<p align="left">
  <img src="https://img.shields.io/github/last-commit/RafaelGallo/MLOps-deploy-credit-card-fraud-detection?style=flat-square" alt="Último commit">
  <img src="https://img.shields.io/github/stars/RafaelGallo/MLOps-deploy-credit-card-fraud-detection?style=flat-square" alt="Stars">
  <img src="https://img.shields.io/github/forks/RafaelGallo/MLOps-deploy-credit-card-fraud-detection?style=flat-square" alt="Forks">
  <img src="https://img.shields.io/github/repo-size/RafaelGallo/MLOps-deploy-credit-card-fraud-detection?style=flat-square" alt="Tamanho do repositório">
  <img src="https://img.shields.io/github/license/RafaelGallo/MLOps-deploy-credit-card-fraud-detection?style=flat-square" alt="Licença">
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-2.2.3-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-2.1.3-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/LightGBM-4.6.0-02A88E?style=flat-square" alt="LightGBM">
  <img src="https://img.shields.io/badge/XGBoost-model-EB5E28?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/imbalanced--learn-SMOTENC-5C6BC0?style=flat-square" alt="imbalanced-learn">
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Databricks-Delta%20Lake-FF3621?style=flat-square&logo=databricks&logoColor=white" alt="Databricks">
  <img src="https://img.shields.io/badge/MLflow-tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Apache%20Airflow-2.10-017CEE?style=flat-square&logo=apacheairflow&logoColor=white" alt="Airflow">
  <img src="https://img.shields.io/badge/Docker-compose-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Kaggle-dataset-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle">
</p>

## Sumário

- [Visão geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Dados](#dados)
- [Modelagem e resultados](#modelagem-e-resultados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Aplicação Streamlit](#aplicação-streamlit)
- [API FastAPI](#api-fastapi)
- [Airflow e agendamento](#airflow-e-agendamento)
- [Deploy online](#deploy-online)
- [Limitações e próximos passos](#limitações-e-próximos-passos)
- [Licença](#licença)

## Visão geral

Fraudes são raras, então a base é extremamente desbalanceada e a acurácia não serve como métrica. O projeto:

1. Processa a base no Databricks em camadas **bronze → silver → gold** (Delta Lake).
2. Treina e compara 8 algoritmos com balanceamento **SMOTENC** (oversampling que respeita variáveis categóricas).
3. Salva cada modelo como um **pipeline completo** (`.joblib`) que já inclui o pré-processamento, então recebe as colunas brutas da transação.
4. Serve o modelo por **Streamlit** (interface) e **FastAPI** (integração).
5. Agenda a verificação diária do modelo e o retreino semanal no **Airflow**.

O modelo padrão em produção é o **Random Forest**, o melhor nas métricas ponderadas.

## Arquitetura

```
Kaggle (11,2 GB)
      │
      ▼
Databricks ── Delta Lake: bronze → silver → gold
      │         └─ treino (SMOTENC) + MLflow
      ▼
models/*.joblib  (pipeline = pré-processamento + modelo)
      │
      ├──► Streamlit  (porta 8501)  interface: transação única e lote CSV
      ├──► FastAPI    (porta 8000)  endpoint /predict
      └──► Airflow    (porta 8080)  checagem diária + retreino semanal
```

## Dados

- **Fonte:** [Credit Card Fraud Mega Dataset (Kaggle)](https://www.kaggle.com/datasets/karthikgangula/credit-card-fraud-mega-dataset/data), cerca de **11,2 GB**.
- **A base não está neste repositório** nem é necessária para o deploy: o modelo salvo já contém o pré-processamento.
- Dados pessoais são removidos na camada silver.

### Features usadas pelo modelo

| Coluna | Tipo | Descrição |
|---|---|---|
| `category` | categórica | Categoria do estabelecimento (14 valores, ex.: `grocery_pos`, `shopping_net`) |
| `state` | categórica | Estado do cliente (51 valores, siglas dos EUA) |
| `gender` | categórica | `F` ou `M` |
| `amt` | numérica | Valor da transação |
| `city_pop` | numérica | População da cidade |
| `age` | numérica | Idade do cliente |
| `distance_km` | numérica | Distância entre cliente e comerciante (km) |
| `hour` | numérica | Hora da transação (0–23) |
| `day_of_week` | numérica | Dia da semana (1 = domingo … 7 = sábado) |
| `month` | numérica | Mês (1–12) |

Alvo: `is_fraud` (1 = fraude).

## Modelagem e resultados

Pipeline de cada modelo: `OrdinalEncoder` (categóricas, categorias desconhecidas → `-1`) + `StandardScaler` (numéricas) + algoritmo. O oversampling com **SMOTENC** (`sampling_strategy=0.5`, `k_neighbors=5`) é aplicado **somente no treino**; o teste permanece intacto, com pesos amostrais para refletir a distribuição real.

Métricas ponderadas no conjunto de teste (`output/metricas_smotenc.csv`):

| Modelo | PR-AUC | ROC-AUC | Recall com precisão ≥ 90% |
|---|---|---|---|
| **random_forest** | **0,702** | 0,971 | **56,7%** |
| adaboost | 0,588 | 0,968 | 45,9% |
| extra_trees | 0,298 | 0,929 | 8,8% |
| xgboost | 0,204 | 0,898 | 7,7% |
| lightgbm | 0,179 | 0,935 | 3,2% |
| hist_gradient_boosting | 0,107 | 0,905 | 2,9% |
| arvore_decisao | 0,011 | 0,784 | 0,0% |
| regressao_logistica | 0,004 | 0,416 | 0,0% |

Operando no limiar que maximiza o F1 (`output/classification_report_smotenc.csv`), o Random Forest atinge **precisão 87,2%**, **recall 59,6%** e **F1 0,708**, com taxa de alarme falso de 0,04%.

> A PR-AUC é a métrica principal porque a classe positiva é muito rara.

## Estrutura do repositório

```
.
├── app/
│   ├── streamlit_app.py        # interface Streamlit
│   └── main.py                 # API FastAPI
├── airflow/
│   ├── dags/
│   │   ├── monitoramento_modelo.py   # checagem diária do modelo
│   │   └── retreino_fraude.py        # retreino semanal (Databricks)
│   └── retrain_lightgbm.py     # script de retreino executado no Databricks
├── docker/
│   ├── Dockerfile              # API FastAPI
│   ├── Dockerfile.streamlit    # app Streamlit
│   ├── Dockerfile.airflow      # Airflow + dependências do modelo
│   └── requirements.txt
├── models/                     # pipelines treinados (.joblib)
├── mlflow/modelos/             # modelos exportados do MLflow
├── output/                     # métricas, relatórios e saúde do modelo
├── docker-compose.yml          # sobe Streamlit + API + Airflow
├── requirements.txt
└── README.md
```

## Como executar

### Pré-requisitos

- Python 3.11 ou 3.12
- Docker Desktop (para a stack completa)

### Opção 1 – Local

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate        # Linux/macOS
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

Acesse http://localhost:8501.

> Use as versões fixadas em `requirements.txt` (`scikit-learn==1.6.1`, `numpy==2.1.3`, `lightgbm==4.6.0`): os modelos foram salvos com elas e versões diferentes podem falhar ao carregar o `.joblib`.

### Opção 2 – Docker Compose (stack completa)

```bash
docker compose up --build
```

| Serviço | URL | Observação |
|---|---|---|
| Streamlit | http://localhost:8501 | interface do modelo |
| API FastAPI | http://localhost:8000/docs | documentação interativa |
| Airflow | http://localhost:8080 | usuário `admin`, senha `admin` |

Para parar: `docker compose down`.

### Opção 3 – Apenas um container

```bash
docker build -f docker/Dockerfile.streamlit -t fraude-streamlit .
docker run -p 8501:8501 fraude-streamlit
```

### Trocar o modelo

O caminho do modelo vem da variável `MODELO_PATH`, sem alterar código:

```bash
# Windows PowerShell
$env:MODELO_PATH = "models/adaboost.joblib"
streamlit run app/streamlit_app.py
```

## Aplicação Streamlit

### Demonstração

Transação classificada como **fraude** (probabilidade de 82,54%):

![Transação classificada como fraude](img/fraud.png)

Transação classificada como **legítima** (probabilidade de 20,72%, abaixo do limiar de 0,50):

![Transação classificada como legítima](img/no_fraud.png)

### Funcionalidades

- **Transação única:** formulário com os campos acima; mostra a probabilidade de fraude e o veredito.
- **Lote (CSV):** envie um CSV com as 10 colunas; o app classifica tudo, mostra o total de fraudes sinalizadas e permite baixar o resultado.
- **Limiar de decisão:** controle na barra lateral (padrão 0,50). Limiar menor aumenta o recall e também os falsos positivos.

## API FastAPI

Variáveis de ambiente: `MODELO_PATH` (caminho do `.joblib`) e `LIMIAR` (padrão `0.5`).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"category":"shopping_net","state":"CA","gender":"F","amt":1200,
       "city_pop":1500,"age":29,"distance_km":85,"hour":2,
       "day_of_week":6,"month":11}'
```

Resposta:

```json
{"probabilidade_fraude": 0.93, "fraude": true, "limiar": 0.5}
```

| Endpoint | Método | Função |
|---|---|---|
| `/health` | GET | checagem de disponibilidade |
| `/predict` | POST | probabilidade e decisão de fraude |

## Airflow e agendamento

| DAG | Agenda | O que faz |
|---|---|---|
| `monitoramento_modelo_fraude` | diária (`@daily`) | Carrega o modelo, pontua transações de exemplo, valida que as probabilidades estão em [0, 1] e grava `output/saude_modelo.json` |
| `retreino_fraude_lightgbm` | semanal (`@weekly`) | Dispara no Databricks o job de ETL e depois o de retreino |

**Retreino com campeão/desafiante:** `retrain_lightgbm.py` treina um desafiante, compara a PR-AUC ponderada com o modelo atual no mesmo teste e só promove se vencer por uma margem de 0,002, guardando backup do campeão anterior e registrando a execução no MLflow.

**Configuração da DAG de retreino:** crie no Airflow a conexão `databricks_default` (host e token do workspace) e troque os `job_id` de exemplo (`111` e `222`) pelos IDs reais dos seus jobs. Sem isso, a DAG semanal falha; a DAG diária de monitoramento funciona sem configuração extra.

> A senha `admin` do Airflow é apenas para uso local. Troque-a antes de expor o serviço.

## Deploy online

**Streamlit Community Cloud (gratuito):**

1. Suba o repositório ao GitHub, incluindo `models/random_forest.joblib`.
2. Em https://share.streamlit.io clique em **Create app**.
3. Escolha o repositório, a branch `main` e o arquivo `app/streamlit_app.py`.
4. Em *Advanced settings*, selecione Python 3.11 ou 3.12 e clique em **Deploy**.

## Limitações e próximos passos

- Os limiares ótimos por modelo (critérios `p90` e `f1`) existem no notebook do Databricks, mas `output/limiares_smotenc.json` está vazio. Exportá-los permitiria fixar o limiar de produção do Random Forest.
- O script de retreino atual cobre o LightGBM; falta um equivalente para o Random Forest, que é o modelo em uso.
- Monitoramento de *data drift* e de desempenho em produção ainda não implementado.
- Autenticação na API e senhas fora do código para um ambiente de produção.

## Licença

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE).

## Autor

**Rafael Gallo**
