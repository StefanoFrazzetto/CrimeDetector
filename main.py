"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from Classification import Benchmark, ClassifierType, MetricType
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

benchmark = Benchmark(dataset)
# benchmark.add_classifier(ClassifierType.RandomForest)
# benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
benchmark.add_classifier(ClassifierType.SupportVectorMachine)
benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)

benchmark.initialize_classifiers()

benchmark.select_metrics(MetricType.ACCURACY,MetricType.AUC)

benchmark.run(10)
# benchmark.get_info()
# benchmark.plot_metrics()
benchmark.plot_metrics()
