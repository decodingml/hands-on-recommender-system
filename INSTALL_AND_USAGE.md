# 🚀 Installation and Usage Guide

This guide will help you set up and run a machine learning pipeline that includes feature engineering, model training, and deployment using Hopsworks and OpenAI.

# 📑 Table of Contents

- [📋 Prerequisites](#-prerequisites)
- [🎯 Getting Started](#-getting-started)
- [⚡️ Running the ML Pipelines](#️-running-the-ml-pipelines)
- [🛠️ Additional Operations](#️-additional-operations)
- [🤖 Running the ML Pipelines in GitHub Actions](#-running-the-ml-pipelines-in-github-actions)
- [🌐 Live Demo](#-live-demo)
- [☁️ Deploying the Streamlit App](#️-deploying-the-streamlit-app)

# 📋 Prerequisites

## Local Tools
You'll need the following tools installed locally:
- [Python v3.11](https://www.python.org/downloads/)
- [uv v0.4.30](https://github.com/astral-sh/uv) - Python package installer and virtual environment manager
- [GNU Make 3.81](https://www.gnu.org/software/make/) - Build automation tool

## Cloud Services
The project requires access to these cloud services:

| Service | Purpose | Cost | Required Credentials | Setup Guide |
|---------|---------|------|---------------------|-------------|
| [Hopsworks](https://rebrand.ly/serverless-github) | AI Lakehouse for feature store, model registry, and serving | Free tier available | `HOPSWORKS_API_KEY` | [Create API Key](https://docs.hopsworks.ai/latest/user_guides/projects/api_key/create_api_key/) |
| [GitHub Actions](https://github.com/features/actions) | Compute & Automation | Free for public repos | - | - |
| [OpenAI API](https://openai.com/index/openai-api/) | LLM API for recommender system | Pay-per-use | `OPENAI_API_KEY`<br>`OPENAI_MODEL_ID` | [Quick Start Guide](https://platform.openai.com/docs/quickstart) |

# 🎯 Getting Started

## 1. Installation

Set up the project environment by running the following:
```bash
make install
```
Test that you have Python 3.11.8 installed in your new `uv` environment:
```bash
uv run python --version
# Output: Python 3.11.8
```

This command will:
- Create a virtual environment using `uv`
- Activate the virtual environment
- Install all dependencies from `pyproject.toml`

> [!NOTE]
> Normally, `uv` will pick the right Python version mentioned in `.python-version` and install it automatically if it is not on your system. If you are having any issues, explicitly install the right Python version by running `make install-python`

## 2. Environment Configuration

Before running any components:
1. Create your environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure the required credentials following the inline comments.

# ⚡️ Running the ML Pipelines

You can run the entire pipeline at once or execute individual components.

## Running the Complete Pipeline

Execute all components in sequence:
```bash
make all
```

This runs the following steps:
1. Feature engineering
2. Retrieval model training
3. Ranking model training
4. Candidate embeddings creation
5. Inference pipeline deployment
6. Materialization job scheduling
7. Inference pipeline for LLM Ranking Predictor

## Running Individual Components

Each component can be run separately:

1. **Feature Engineering**
   ```bash
   make feature-engineering
   ```
   View results in [Hopsworks Serverless](https://rebrand.ly/serverless-github): **Feature Store → Feature Groups**

2. **Retrieval Model Training**
   ```bash
   make train-retrieval
   ```
   View results in [Hopsworks Serverless](https://rebrand.ly/serverless-github): **Data Science → Model Registry**

3. **Ranking Model Training**
   ```bash
   make train-ranking
   ```
   View results in [Hopsworks Serverless](https://rebrand.ly/serverless-github): **Data Science → Model Registry**

4. **Embeddings Creation**
   ```bash
   make create-embeddings
   ```
   View results in [Hopsworks Serverless](https://rebrand.ly/serverless-github): **Feature Store → Feature Groups**

5. **Deployment Creation**
   ```bash
   make create-deployments
   ```
   View results in [Hopsworks Serverless](https://rebrand.ly/serverless-github): **Data Science → Deployments**

<p align="center">
  <a href="https://rebrand.ly/serverless-github">
    <img src="assets/hopsworks_deployments.png" alt="hopsworks_deployments" width="800">
  </a>
</p>

6. **Materialization Job Scheduling**
   ```bash
   make schedule-materialization-jobs
   ```
   View results in [Hopsworks Serverless](https://rebrand.ly/serverless-github): **Compute → Ingestions**

7. **Deploy LLM Predictor**
   ```bash
   make deploy-llm-predictor
   ```
   View results in [Hopsworks Serverless](https://rebrand.ly/serverless-github): **Compute → Ingestions**

deploy-llm-predictor

## 🚨 Important Notes
- All notebooks are executed using IPython through the UV virtual environment
- Components should be run in the specified order when executing individually
- Ensure UV is properly installed and configured before running any commands

# 🛠️ Additional Operations

## LLM Ranking Deployment
If you want to use the LLM Ranker instead of the trained ranker, make sure to have the right custom environment configured in Hopsworks, on which you should install langchain and langchain-openai.
In order to configure it, follow the next steps:
1. In Hopsworks, go to Project Settings -> Python Environments
2. Choose an environemnt from the Inference Pipeline, for example pandas-inference-pipeline and select it
3. Clone Env -> Choose a suitable name for the custom env -> Create -> Select the new environment
4.  Install Libraries -> With 'Pip' and 'Search' selected -> Find by name 'langchain' -> Choose from dropdown 0.2.6 version -> Press +
5. Install Libraries -> With 'Pip' and 'Name/version' selected -> Library name = 'langchain-openai' and version = '0.1.14' -> Install

## Launch Frontend Application
Start the Streamlit UI that interfaces with [Hopsworks](https://rebrand.ly/serverless-github):
```bash
make start-ui
```
Accessible at `http://localhost:8501/`

> [!IMPORTANT]
> If you get `ModelServingException` or `ConnectionError` errors, the instances are still scaled to 0, so give it a few minutes to scale up. Then, refresh the page. This happens because we are in demo, 0-cost mode.

## Clean Up Resources
Remove all created resources from [Hopsworks Serverless](https://rebrand.ly/serverless-github):
```bash
make clean-hopsworks-resources
```

# 🤖 Running the ML Pipelines in GitHub Actions

This project supports running ML pipelines automatically through GitHub Actions, providing an alternative to local or Colab execution.

> [!NOTE]
> This is handy when getting network errors, such as timeouts, on your local machine. GitHub Actions has an enterprise-level network that will run your ML pipelines smoothly.

## Pipeline Triggers

The ML pipelines can be triggered in three ways:
- Manual trigger through GitHub UI
- Scheduled execution (configurable)
- On push to main branch (configurable)

## Setup Process

### 1. Fork Repository
Create your own copy of the repository to access GitHub Actions:
```bash
# Use GitHub's UI to fork the repository
https://github.com/original-repo/name → Your-Username/name
```
[📚 GitHub Fork Guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)

### 2. Configure Secrets
Set up required environment variables as GitHub Actions secrets:

**Option A: Using GitHub UI**
1. Navigate to: Repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add required secrets:
   - `HOPSWORKS_API_KEY`
   - `OPENAI_API_KEY`

[📚 Set up GitHub Actions Secrets Guide](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions?tool=webui)

<p align="center">
  <a href="https://decodingml.substack.com/p/33d3273e-b8e3-4d98-b160-c3d239343022">
    <img src="assets/github_actions_secrets.png" alt="GA Secrets" width="800">
  </a>
</p>

**Option B: Using GitHub CLI**

If you have `GitHub CLI` installed, instead of settings the GitHub Actions secrets manually, you can set them by running the following:

```bash
gh secret set HOPSWORKS_API_KEY
gh secret set OPENAI_API_KEY
```

### 3. Execute Pipeline

#### Manual Execution
1. Go to Actions → ML Pipelines
2. Click "Run workflow"
3. Select branch (default: main)
4. Click "Run workflow"

<p align="center">
  <a href="https://decodingml.substack.com/p/33d3273e-b8e3-4d98-b160-c3d239343022">
    <img src="assets/github_actions_manual_trigger.png" alt="GA Manual Trigger" width="1000">
  </a>
</p>

After triggering the pipeline, you will see it running, signaled by a yellow circle. Click on it to see the progress.

<p align="center">
  <a href="https://decodingml.substack.com/p/33d3273e-b8e3-4d98-b160-c3d239343022">
    <img src="assets/github_actions_pipeline_progress.png" alt="GA Progress" width="1000">
  </a>
</p>

After it is finished, it should look like this:

<p align="center">
  <a href="https://decodingml.substack.com/p/33d3273e-b8e3-4d98-b160-c3d239343022">
    <img src="assets/github_actions_pipeline_done.png" alt="GA Done" width="1000">
  </a>
</p>

#### Automated Execution

Another option is to run the ML pipelines automatically on a schedule or when new commits are pushed to the main branch.

Edit `.github/workflows/ml_pipelines.yaml` to enable automatic triggers:

```yaml
name: ML Pipelines

on:
  # schedule: # Uncomment to run the pipelines every 2 hours. All the pipelines take ~1.5 hours to run.
  #   - cron: '0 */2 * * *'
  # push: # Uncomment to run pipelines on every new commit to main
  #   branches:
  #     - main
  workflow_dispatch:  # Allows manual triggering from GitHub UI
```

## Monitoring & Results

1. **Pipeline Progress**
   - View real-time execution in Actions tab
   - Each step shows detailed logs and status

2. **Output Verification**
   - Access results in [Hopsworks Serverless](https://rebrand.ly/serverless-github)
   - Check Feature Groups, Feature Views, Model Registry, and Deployments

## ⚠️ Important Notes
- Full pipeline execution takes approximately 1.5 hours
- Ensure sufficient GitHub Actions minutes available
- Monitor usage when enabling automated triggers

# 🌐 Live Demo

Try out our deployed H&M real-time personalized recommender:
[💻 Live Streamlit Demo](https://decodingml-hands-on-personalized-recommender.streamlit.app/)

> [!IMPORTANT]
> If you get `ModelServingException` or `ConnectionError` errors, the instances are still scaled to 0, so give it a few minutes to scale up. Then, refresh the page. This happens because we are in demo, 0-cost mode

# ☁️ Deploying the Streamlit App

Deploying a Streamlit App to their [cloud](https://streamlit.io/cloud) is free and straightforward after the GitHub repository is set in right place:

- `uv.lock` - installing Python dependencies
- `packages.txt` - installing system dependencies
- `streamlit_app.py` - entrypoint to the Streamlit application

## Deployment Steps

### 1. Repository Setup
Fork the repository if you haven't already:
```bash
# Use GitHub's UI to fork the repository
https://github.com/original-repo/name → Your-Username/name
```
[📚 GitHub Fork Guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)

### 2. Streamlit Cloud Setup
1. Create a free account on [Streamlit Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started)
2. Navigate to [New App Deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
3. Configure deployment settings:

| Setting | Configuration | Description |
|---------|--------------|-------------|
| App Type | ![App Type](assets/streamlit_choose_app_type.png) | Select "Deploy a public app from GitHub" |
| Main Settings | ![Main Settings](assets/streamlit_choose_main_settings.png) | Configure your repository |
| Advanced Settings | ![Advanced Settings](assets/streamlit_choose_advanced_settings.png) | Set Python 3.11 and `HOPSWORKS_API_KEY` |

## ⚠️ Important Notes
- Ensure all required files are present in your repository
- Python version must be set to 3.11
- `HOPSWORKS_API_KEY` must be configured in environment variables
- Repository must be public for free tier deployment

[📚 More on Streamlit Cloud deployments](https://docs.streamlit.io/deploy)