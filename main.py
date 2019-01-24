"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""
from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from Classification import Benchmark, ClassifierType
from Data import Dataset
from PreProcessing import CorpusName, CorpusParser

base_path = "/home/stefano/Documents/University/DissertationDatasets"
pan12_dir = f"{base_path}/pan12-sexual-predator-identification-test-corpus-2012-05-21"

parser = CorpusParser.factory(CorpusName.PAN12, merge_messages=False)
parser.set_source_directory(pan12_dir)

if parser.is_serialized():
    parser = parser.deserialize()
else:
    parser.parse()
    parser.serialize()

parser.log_info()

dataset = Dataset(0.85)
if dataset.is_serialized():
    dataset = dataset.deserialize()
else:
    dataset = parser.get_dataset()

dataset.finalize()
dataset.log_info()
dataset.balance_negatives()
dataset.log_info()

training_data = dataset.training['text']
training_labels = dataset.training['label']

testing_data = dataset.validation['text']
testing_labels = dataset.validation['label']

feature_extraction = CountVectorizer(
    # analyzer='word',
    # max_features=10,
    # ngram_range=(2,3),
    # stop_words='english',
    # max_df=0.5
)

training_vectors = feature_extraction.fit_transform(training_data)
testing_vectors = feature_extraction.transform(testing_data)


tfidf = TfidfTransformer()
training_vectors = tfidf.fit_transform(training_vectors, training_labels)
testing_vectors = tfidf.transform(testing_vectors)

benchmark = Benchmark(dataset)
benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
benchmark.add_classifier(ClassifierType.SupportVectorMachine)
benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)

benchmark.initialize_classifiers(training_vectors, training_labels)
benchmark.run(testing_vectors, testing_labels)
benchmark.plot_metrics()
