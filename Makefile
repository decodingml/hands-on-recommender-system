install-python:
	uv python install

install:
	uv venv
	. .venv/bin/activate
	uv pip install --all-extras --requirement pyproject.toml

start-ui:
	uv run python -m streamlit run streamlit_app.py

clean-hopsworks-resources:
	uv run python tools/clean_hopsworks_resources.py

all: feature-engineering train-retrieval train-ranking create-embeddings create-deployments deploy-llm-predictor schedule-materialization-jobs

feature-engineering:
	uv run ipython notebooks/1_fp_computing_features.ipynb

train-retrieval:
	uv run ipython notebooks/2_tp_training_retrieval_model.ipynb

train-ranking:
	uv run ipython notebooks/3_tp_training_ranking_model.ipynb

create-embeddings:
	uv run ipython notebooks/4_ip_computing_item_embeddings.ipynb

create-deployments:
	uv run ipython notebooks/5_ip_creating_deployments.ipynb

deploy-llm-predictor:
	uv run ipython notebooks/7_ip_creating_llm_ranking_deployment.ipynb

schedule-materialization-jobs:
	uv run ipython notebooks/6_scheduling_materialization_jobs.ipynb