"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""
import sys
from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer

from Classification import Benchmark, ClassifierType
from Data import Dataset
from Interfaces import Analyzable
from PreProcessing import CorpusName, CorpusParser
from Utils import Log, DataConverter

base_path = "/home/stefano/Documents/University/DissertationDatasets"
pan12_dir = f"{base_path}/pan12-sexual-predator-identification-test-corpus-2012-05-21"

parser = CorpusParser.factory(CorpusName.PAN12)
parser.set_source_directory(pan12_dir)

if parser.is_serialized():
    parser = parser.deserialize()
else:
    parser.parse()
    parser.serialize()

parser.log_info()

dataset = Dataset()
if dataset.is_serialized():
    dataset = dataset.deserialize()
else:
    dataset = parser.get_dataset()
    dataset.finalize()
    dataset.under_sample(0.7)
    dataset.log_info()
    # dataset.under_sample()
    # dataset.log_info()

# (dataset.training)

# sys.exit(1)

training_data = dataset.training['text']
training_labels = dataset.training['label']

testing_data = dataset.testing['text']
testing_labels = dataset.testing['label']

feature_extraction = CountVectorizer()
training_vectors = feature_extraction.fit_transform(training_data)
testing_vectors = feature_extraction.transform(testing_data)

# Log.info(f"Training vectors length: {len(training_vectors)}")

benchmark = Benchmark(dataset)
# benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
benchmark.add_classifier(ClassifierType.SupportVectorMachine)
# benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)

benchmark.initialize_classifiers(training_vectors, training_labels)
benchmark.run(testing_vectors, testing_labels)
benchmark.plot_metrics()
