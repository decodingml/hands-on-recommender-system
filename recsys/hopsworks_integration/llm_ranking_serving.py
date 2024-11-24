import os

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
    def deploy(cls, project):
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
            transformer=ranking_transformer
        )

        return ranking_deployment
