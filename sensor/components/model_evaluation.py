from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.entity.artifact_entity import DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel, ModelResolver, TargetValueMapping
from sensor.utils.main_utils import load_object, write_yaml_file, save_object

import os
import sys
import pandas as pd

class ModelEvaluation:

    def __init__(self, model_trainer_artifact: ModelTrainerArtifact,
                        data_validation_artifact: DataValidationArtifact,
                        model_evaluation_config: ModelEvaluationConfig):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise SensorException(e,sys)

    def initiate_model_evaluation(self):
        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            # read the file into dataframe
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            # concatenate the train and test data to make into one single dataframe
            df = pd.concat([train_df, test_df])

            # get the target feature separated
            y_true = df[TARGET_COLUMN]
            y_true.replace(TargetValueMapping().to_dict(), inplace=True)
            df.drop(TARGET_COLUMN, axis=1, inplace=True)

            # load the model
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True

            # if model doesn't exist, return empty on accuracy and best model as this will be the first model.
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=None,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None
                )
                logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            # Get the latest model and load it. Load the trained model as well.
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            # predict for both models
            y_latest_pred = latest_model.predict(df)
            y_trained_pred = latest_model.predict(df)

            # Get the classification metrics for the two models
            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            #Check if there is an improvement in the trained model compared to the latest one.
            # If yes, then accept the model
            improved_accuracy = trained_metric.f1_score - latest_metric.f1_score
            if improved_accuracy > self.model_evaluation_config.change_threshold:
                is_model_accepted = True
            else:
                is_model_accepted = False

            # Prepare the artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric
            )
            
            # Prepare the model evaluation report
            model_evaluation_report = model_evaluation_artifact.__dict__
        
            # Save the report
            write_yaml_file(file_path=self.model_evaluation_config.report_file_path, content=model_evaluation_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise SensorException(e,sys)