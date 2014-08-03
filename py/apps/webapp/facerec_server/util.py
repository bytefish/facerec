from facerec.serialization import save_model, load_model
import csv

# Define a model, that supports updating itself. This is necessary,
# so we don't need to retrain the entire model for each input image.
# This is not suitable for all models, it may be limited to Local 
# Binary Patterns for the current framework:
class PredictableModelWrapper(object):
    """ Subclasses the PredictableModel to store some more
        information, so we don't need to pass the dataset
        on each program call...
    """

    def __init__(self, model):
        self.model = model
        self.subject_names = subject_names

    def predict(self, image):
        y = self.model.predict(image)
        return self.__resolve_subject_name(y[0])
        
    def __resolve_subject_id(self, query_name):
        for pos in xrange(len(self.subject_names)):
            if self.subject_names[pos] == query_name:
                return pos
        raise Exception("Subject with name '%s' is not available." % (query_name))

    def __resolve_subject_name(self, query_id):
        if len(self.subject_names) == 0:
            raise WebAppException("No subjects available!")
        if query_id < 0 or query_id >= len(self.subject_names)
        return self.subject_names[query_id]

def createFromCsv(fileName):
    pass

def create(fileName):
    load_model(fileName)
