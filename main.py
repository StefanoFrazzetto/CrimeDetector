"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""

from Classification import Benchmark, ClassifierType, MetricType
from Data import Dataset
from PreProcessing import CorpusName, CorpusParser

base_path = "/home/stefano/Documents/University/DissertationDatasets"
pan12_dir = f"{base_path}/pan12-sexual-predator-identification-test-corpus-2012-05-21"

parser = CorpusParser.factory(CorpusName.PAN12, merge_messages=False)
parser.set_source_directory(pan12_dir)
dataset = Dataset(CorpusName.PAN12)

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
benchmark.add_classifier(ClassifierType.RandomForest)
benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
benchmark.add_classifier(ClassifierType.SupportVectorMachine)
benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)
benchmark.add_classifier(ClassifierType.LogisticRegression)
benchmark.initialize_classifiers()


benchmark.select_metrics(
    MetricType.ACCURACY,
    MetricType.PRECISION,
    MetricType.RECALL,
    MetricType.ROC
)
benchmark.run(10)
# benchmark.get_info()
# benchmark.save_metrics('./results')
benchmark.plot_metrics()
# benchmark.clustering()
