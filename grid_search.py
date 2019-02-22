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

base_path = "/home/stefano/Documents/University/DissertationDatasets"
pan12_dir = f"{base_path}/pan12-sexual-predator-identification-test-corpus-2012-05-21"

parser = CorpusParser.factory(CorpusName.PAN12, pan12_dir, merge_messages=False)
dataset = Dataset(parser.get_params(), CorpusName.PAN12)

# If the dataset for this corpus is already serialized
if dataset.is_serialized():
    dataset = dataset.deserialize()
    dataset.log_info()

# otherwise create it using the parser
else:
    if parser.is_serialized():
        parser = parser.deserialize()
    else:
        parser.parse()
        parser.serialize()

    parser.log_info()

    dataset = parser.add_to_dataset()
    dataset.finalize()
    dataset.log_info()
    dataset.autobalance()
    dataset.log_info()
    dataset.serialize()

training_data = dataset.training['text']
training_labels = dataset.training['label']

testing_data = dataset.validation['text']
testing_labels = dataset.validation['label']

grid_search = GridSearch(ClassifierType.LogisticRegression)
grid_search.fit(training_data, training_labels)
