import tensorflow as tf
from hsml.model_schema import ModelSchema
from hsml.schema import Schema

from recsys.models.two_tower import ItemTower, QueryTower


class QueryModelModule(tf.Module):
    def __init__(self, query_model: QueryTower):
        self.query_model = query_model

    @tf.function()
    def compute_emb(self, instances):
        query_emb = self.query_model(instances)

        return {
            "customer_id": instances["customer_id"],
            "month_sin": instances["month_sin"],
            "month_cos": instances["month_cos"],
            "query_emb": query_emb,
        }

    def save_local(self, output_path: str = "query_model") -> str:
        # Define the input specifications for the instances
        instances_spec = {
            "customer_id": tf.TensorSpec(
                shape=(None,), dtype=tf.string, name="customer_id"
            ),  # Specification for customer IDs
            "month_sin": tf.TensorSpec(
                shape=(None,), dtype=tf.float64, name="month_sin"
            ),  # Specification for sine of month
            "month_cos": tf.TensorSpec(
                shape=(None,), dtype=tf.float64, name="month_cos"
            ),  # Specification for cosine of month
            "age": tf.TensorSpec(
                shape=(None,), dtype=tf.float64, name="age"
            ),  # Specification for age
        }

        # Get the concrete function for the query_model's compute_emb function using the specified input signatures
        signatures = self.compute_emb.get_concrete_function(instances_spec)

        # Save the query_model along with the concrete function signatures
        tf.saved_model.save(
            self.query_model,  # The model to save
            output_path,  # Path to save the model
            signatures=signatures,  # Concrete function signatures to include
        )

        return output_path

    def save_hopsworks(self, mr, query_df, emb_dim) -> None:
        local_model_path = self.save_local()

        # Each model needs to be set up with a
        # [Model Schema](https://docs.hopsworks.ai/machine-learning-api/latest/generated/model_schema/),
        # which describes the inputs and outputs for a model.
        # A schema can either be manually specified or inferred from data.

        # Infer input schema from data.
        query_model_input_schema = Schema(query_df)
        # Manually specify output schema.
        query_model_output_schema = Schema(
            [
                {
                    "name": "query_embedding",
                    "type": "float32",
                    "shape": [emb_dim],
                }
            ]
        )
        query_model_schema = ModelSchema(
            input_schema=query_model_input_schema,
            output_schema=query_model_output_schema,
        )

        # Sample a query example from the query DataFrame
        query_example = query_df.sample().to_dict("records")

        # Create a tensorflow model for the query_model in the Model Registry
        mr_query_model = mr.tensorflow.create_model(
            name="query_model",  # Name of the model
            description="Model that generates query embeddings from user and transaction features",  # Description of the model
            input_example=query_example,  # Example input for the model
            model_schema=query_model_schema,  # Schema of the model
        )

        # With the schema in place, you can finally register your model.
        # Save the query_model to the Model Registry
        mr_query_model.save(local_model_path)  # Path to save the model


class CandidateModelModule(tf.Module):
    def __init__(self, item_model: ItemTower):
        self.item_model = item_model

    def save_local(self, output_path: str = "candidate_model") -> str:
        tf.saved_model.save(
            self.item_model,  # The model to save
            output_path,  # Path to save the model
        )

        return output_path
    
    def save_hopsworks(self, mr, item_df, emb_dim):
        local_model_path = self.save_local()

        # Define the input schema for the candidate_model based on item_df
        candidate_model_input_schema = Schema(item_df)

        # Define the output schema for the candidate_model, specifying the shape and type of the output
        candidate_model_output_schema = Schema(
            [
                {
                    "name": "candidate_embedding",  # Name of the output feature
                    "type": "float32",  # Data type of the output feature
                    "shape": [emb_dim],  # Shape of the output feature
                }
            ]
        )

        # Combine the input and output schemas to create the overall model schema for the candidate_model
        candidate_model_schema = ModelSchema(
            input_schema=candidate_model_input_schema,  # Input schema for the model
            output_schema=candidate_model_output_schema,  # Output schema for the model
        )

        # Sample a candidate example from the item DataFrame
        candidate_example = item_df.sample().to_dict("records")

        # Create a tensorflow model for the candidate_model in the Model Registry
        mr_candidate_model = mr.tensorflow.create_model(
            name="candidate_model",  # Name of the model
            description="Model that generates candidate embeddings from item features",  # Description of the model
            input_example=candidate_example,  # Example input for the model
            model_schema=candidate_model_schema,  # Schema of the model
        )

        # Save the candidate_model to the Model Registry
        mr_candidate_model.save(local_model_path)  # Path to save the model