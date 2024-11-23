import logging

from langchain import PromptTemplate, LLMChain
from langchain_openai import ChatOpenAI

from recsys.config import settings


class LLMPredict(object):
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY

        self.input_variables = ["age", "month_sin", "month_cos", "product_type_name", "product_group_name",
                                "graphical_appearance_name", "colour_group_name", "perceived_colour_value_name",
                                "perceived_colour_master_name", "department_name", "index_name", "index_group_name",
                                "section_name", "garment_group_name"]
        self.llm = self._build_lang_chain()

    def predict(self, inputs):
        logging.info(f"✅ Inputs: {inputs}")

        # Extract ranking features and article IDs from the inputs
        features = inputs[0].pop("ranking_features")
        article_ids = inputs[0].pop("article_ids")

        # Preprocess features for OpenAI model input
        preprocessed_features = self._preprocess_features(features)
        logging.info(f"predict -> Preprocessed features: {preprocessed_features}")
        logging.info(f"Article IDs: {article_ids}")

        logging.info(f"🦅 Predicting with OpenAI model for {len(features)} instances")

        scores = []
        for feature in preprocessed_features:
            langchain_output = self.llm.invoke(feature)
            scores.append(self._postprocess_output(langchain_output))

        logging.info(f"LLM Scores: {scores}")

        return {
            "scores": scores,
            "article_ids": article_ids,
        }

    def _postprocess_output(self, output):
        return float(output['text'].split(':')[1].strip())

    def _preprocess_features(self, features):
        """
        Convert ranking features into a natural language description
        suitable for OpenAI model input.
        """
        preprocessed = []
        for feature_set in features:
            # Example: Create a descriptive string for each feature set
            query_parameters = {}
            for key, value in zip(self.input_variables, feature_set):
                query_parameters[key] = value
            preprocessed.append(query_parameters)
        return preprocessed

    def _build_lang_chain(self):
        template = """
            You are a helpful assistant specialized in predicting customer behavior. Your task is to analyze the features of a product and predict the probability of it being purchased by a customer.

            ### Instructions:
            1. Use the provided features of the product to make your prediction.
            2. Consider the following numeric and categorical features:
               - Numeric features: These are quantitative attributes, such as numerical identifiers or measurements.
               - Categorical features: These describe qualitative aspects, like product category, color, and material.
            3. Your response should only include the probability of purchase for the positive class (e.g., likelihood of being purchased), as a value between 0 and 1.

            ### Product Features:
            Numeric features:
            - Age: {age}
            - Month Sin: {month_sin}
            - Month Cos: {month_cos}

            Categorical features:
            - Product Type: {product_type_name}
            - Product Group: {product_group_name}
            - Graphical Appearance: {graphical_appearance_name}
            - Colour Group: {colour_group_name}
            - Perceived Colour Value: {perceived_colour_value_name}
            - Perceived Colour Master Value: {perceived_colour_master_name}
            - Department Name: {department_name}
            - Index Name: {index_name}
            - Department: {index_group_name}
            - Sub-Department: {section_name}
            - Group: {garment_group_name}

            ### Your Task:
            Based on the features provided, predict the probability that the customer will purchase this product to 4-decimals precision. Provide the output in the following format:
            Probability: 
        """

        model = ChatOpenAI(
            model_name='gpt-4o-mini-2024-07-18',
            temperature=0.7,
            openai_api_key=self.openai_api_key,
        )
        prompt = PromptTemplate(
            input_variables=self.input_variables,
            template=template,
        )
        langchain = LLMChain(
            llm=model,
            prompt=prompt,
            verbose=True
        )
        return langchain
