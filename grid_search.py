"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""

from Classification import ClassifierType
from Classification.GridSearch import GridSearch
from Data import Dataset
from PreProcessing import CorpusName, CorpusParser

DATASETS_PATH = "./datasets"
PAN12_PATH = f"{DATASETS_PATH}/pan12"
FORMSPRING_FILE_PATH = f"{DATASETS_PATH}/formspring/formspring_data.csv"


parser = CorpusParser.factory(CorpusName.PAN12, PAN12_PATH, merge_messages=False)
dataset = Dataset(parser.get_params(), CorpusName.PAN12)

if dataset.is_serialized():
    dataset = dataset.deserialize()
    dataset.log_info()
else:
    if parser.is_serialized():
        parser = parser.deserialize()
    else:
        parser.parse()
        parser.serialize()

    parser.log_info()
    parser.add_to_dataset(dataset)

    dataset.finalize()
    dataset.log_info()
    dataset.serialize()

dataset.balance_training(5)
dataset.balance_testing(5)

training_data = dataset.training['data']
training_labels = dataset.training['label']

testing_data = dataset.testing['data']
testing_labels = dataset.testing['label']

grid_search = GridSearch(ClassifierType.SupportVectorMachine)
grid_search.fit(training_data, training_labels, n_jobs=-1)
