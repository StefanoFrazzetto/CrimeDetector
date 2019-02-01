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
from sklearn.svm import SVC

from Classification import Benchmark, ClassifierType, Classifier
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

validation_data = dataset.validation['text']
validation_labels = dataset.validation['label']

count_vectorizer = CountVectorizer(
    # analyzer='word',
    # max_features=10,
    # ngram_range=(2,3),
    # stop_words='english',
    # max_df=0.5
)

training_vectors = count_vectorizer.fit_transform(training_data)
validation_vectors = count_vectorizer.transform(validation_data)


tfidf = TfidfTransformer(
    norm='l2',  # cosine normalization
)
training_vectors = tfidf.fit_transform(training_vectors, training_labels)
testing_vectors = tfidf.transform(validation_vectors)

# tfidf = TfidfVectorizer(count_vectorizer)
# training_vectors = tfidf.fit_transform(training_data, training_labels)
# validation_vectors = tfidf.transform(validation_data)

# svc = SVC(probability=True)
#
# svc.fit(training_vectors, training_labels)
#
# example = [""]
# example = count_vectorizer.transform(example)
# example = tfidf.transform(example)
#
# result = svc.predict_proba(example)
# for k, v in result:
#     pprint(f"{k}: {v}")

benchmark = Benchmark(dataset)
benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
# benchmark.add_classifier(ClassifierType.SupportVectorMachine)
# benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)
#
benchmark.initialize_classifiers(training_vectors, training_labels)
benchmark.run(validation_vectors, validation_labels)
benchmark.plot_metrics()
