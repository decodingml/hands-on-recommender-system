import os
import hopsworks

from hsml.transformer import Transformer
from recsys.config import settings


class HopsworksLLMRankingModel:
    deployment_name = "llmranking"

    @classmethod
    def register(cls, mr):
        local_model_path = str(settings.RECSYS_DIR / "inference" / "llm_ranking_predictor.py")
        ranking_model = mr.python.create_model(
            name="llm_ranking_model",
            description="LLM Ranking model that scores item candidates",
        )
        ranking_model.save(local_model_path)

    @classmethod
    def deploy(cls):
        # Prepare secrets used in the deployment
        cls._prepare_secrets()

        project = hopsworks.login()
        # todo call prep environment once hopsworks upgrade
        # cls._prepare_environment(project)
        mr = project.get_model_registry()
        dataset_api = project.get_dataset_api()

        ranking_model = mr.get_model(
            name="llm_ranking_model"
        )

        # Copy transformer file into Hopsworks File System
        uploaded_file_path = dataset_api.upload(
            str(
                settings.RECSYS_DIR / "inference" / "ranking_transformer.py"
            ),  # File name to be uploaded
            "Resources",  # Destination directory in Hopsworks File System
            overwrite=True,  # Overwrite the file if it already exists
        )
        # Construct the path to the uploaded transformer script
        transformer_script_path = os.path.join(
            "/Projects",  # Root directory for projects in Hopsworks
            project.name,  # Name of the current project
            uploaded_file_path,  # Path to the uploaded file within the project
        )

        # Upload llm predictor file to Hopsworks
        uploaded_file_path = dataset_api.upload(
            str(settings.RECSYS_DIR / "inference" / "llm_ranking_predictor.py"),
            "Resources",
            overwrite=True,
        )

        # Construct the path to the uploaded script
        predictor_script_path = os.path.join(
            "/Projects",
            project.name,
            uploaded_file_path,
        )

        ranking_transformer = Transformer(
            script_file=transformer_script_path,
            resources={"num_instances": 0},
        )

        # Deploy ranking model
        ranking_deployment = ranking_model.deploy(
            name=cls.deployment_name,
            description="Deployment that search for item candidates and scores them based on customer metadata using "
                        "GPT 4",
            script_file=predictor_script_path,
            resources={"num_instances": 0},
            transformer=ranking_transformer,
            environment=settings.CUSTOM_HOPSWORKS_INFERENCE_ENV
        )

        return ranking_deployment

    @classmethod
    def _prepare_environment(cls, project):
        # Upload requirements file to Hopsworks
        dataset_api = project.get_dataset_api()

        requirements_path = dataset_api.upload(
            str(settings.RECSYS_DIR / "hopsworks_integration" / "llm_ranker" / "requirements.txt"),
            "Resources",
            overwrite=True,
        )

        # Check if custom env exists, if not create it
        env_api = project.get_environment_api()
        envs = env_api.get_environments()
        existing_envs = [env.name for env in envs]
        if settings.CUSTOM_HOPSWORKS_INFERENCE_ENV in existing_envs:
            env = env_api.get_environment(settings.CUSTOM_HOPSWORKS_INFERENCE_ENV)
        else:
            env = env_api.create_environment(name=settings.CUSTOM_HOPSWORKS_INFERENCE_ENV)

        # Install the extra requirements in the Python environment on Hopsworks
        env.install_requirements(requirements_path)

    @classmethod
    def _prepare_secrets(cls):
        connection = hopsworks.connection(host="c.app.hopsworks.ai",
                                          hostname_verification=False,
                                          port=443,
                                          api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value()
                                          )
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "Missing required secret: 'OPENAI_API_KEY'. Please ensure it is set in the .env file or config.py "
                "settings.")

        secrets_api = connection.get_secrets_api()
        secrets = secrets_api.get_secrets()
        existing_secret_keys = [secret.name for secret in secrets]
        # Create the OPENAI_API_KEY secret if it doesn't exist
        if "OPENAI_API_KEY" not in existing_secret_keys:
            secrets_api.create_secret(
                "OPENAI_API_KEY",
                settings.OPENAI_API_KEY.get_secret_value(),
                project=settings.HOPSWORKS_PROJECT
            )
