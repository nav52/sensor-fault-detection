from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.config_entity import ModelPusherConfig
from sensor.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
import os
import sys
import shutil

class ModelPusher:

    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                        model_pusher_config: ModelPusherConfig):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise SensorException(e,sys)

    def initiate_model_pusher(self):
        try:
            trained_model_path = self.model_evaluation_artifact.trained_model_path

            # Copy trained model to model pusher's model file path and saved model path
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            shutil.copy(src=trained_model_path, dest=model_file_path)

            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            shutil.copy(src=trained_model_path, dest=saved_model_path)

            # Prepare the artifact
            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path,
                model_file_path=model_file_path
            )
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e,sys)