class TargetValueMapping:
    def __init__(self):
        self.neg: int=0
        self.pos: int=1
    
    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

# Code to train model and check accuracy

class SensorModel:

    def __init__(self):
        pass

    def get_best_model(self):
        pass