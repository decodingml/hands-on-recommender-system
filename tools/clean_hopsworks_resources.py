import hopsworks

# Login to Hopsworks
project = hopsworks.login()


# Get deployment registry
mr = project.get_model_serving()

# List all deployments
deployments = mr.get_deployments()

# Delete each deployment
for deployment in deployments:
    print(f"Deleting deployment: {deployment.name}.")
    deployment.delete()

# Get the model registry
mr = project.get_model_registry()

# List all models
for model_name in ["llm_ranking_model", "ranking_model", "candidate_model", "query_model"]:
    models = mr.get_models(name=model_name)

    # Delete each model
    for model in models:
        print(f"Deleting model: {model.name} (version: {model.version})")
        model.delete()

# Get feature store
fs = project.get_feature_store()

for feature_view in [
    "retrieval",
    "articles",
    "customers",
    "candidate_embeddings",
    "ranking",
]:
    # Get all feature views
    try:
        feature_views = fs.get_feature_views(name=feature_view)
    except:
        print(f"Couldn't find feature view: {feature_view}. Skipping...")
        feature_views = []

    # Delete each feature view
    for fv in feature_views:
        print(f"Deleting feature view: {fv.name} (version: {fv.version})")
        try:
            fv.delete()
        except Exception:
            print(f"Failed to delete feature view {fv.name}.")

for feature_group in [
    "customers",
    "articles",
    "transactions",
    "interactions",
    "candidate_embeddings",
    "ranking",
]:
    # Get all feature groups
    try:
        feature_groups = fs.get_feature_groups(name=feature_group)
    except:
        print(f"Couldn't find feature group: {feature_view}. Skipping...")
        feature_groups = []

    # Delete each feature group
    for fg in feature_groups:
        print(f"Deleting feature group: {fg.name} (version: {fg.version})")
        try:
            fg.delete()
        except:
            print(f"Failed to delete feature group {fv.name}.")
