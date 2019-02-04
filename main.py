"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from Classification import Benchmark, ClassifierType
from Data import Dataset
from PreProcessing import CorpusName, CorpusParser

base_path = "/home/stefano/Documents/University/DissertationDatasets"
pan12_dir = f"{base_path}/pan12-sexual-predator-identification-test-corpus-2012-05-21"

parser = CorpusParser.factory(CorpusName.PAN12, merge_messages=False)
parser.set_source_directory(pan12_dir)
dataset = Dataset(parser.__hash__())

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

    dataset = parser.get_dataset()
    dataset.finalize()
    dataset.log_info()
    dataset.balance_negatives()
    dataset.log_info()
    dataset.serialize()

training_data = dataset.training['text']
training_labels = dataset.training['label']

testing_data = dataset.validation['text']
testing_labels = dataset.validation['label']

count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

training_vectors = count_vectorizer.fit_transform(training_data, training_labels)
testing_vectors = count_vectorizer.transform(testing_data)

training_vectors = tfidf_transformer.fit_transform(training_vectors, training_labels)
testing_vectors = tfidf_transformer.transform(testing_vectors)


benchmark = Benchmark(dataset)
benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
benchmark.add_classifier(ClassifierType.SupportVectorMachine)
benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)
benchmark.add_classifier(ClassifierType.RandomForest)

benchmark.initialize_classifiers()
benchmark.run(10)
# benchmark.plot_metrics('./results')
benchmark.plot_metrics()
