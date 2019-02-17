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
from Utils import Log
from Utils.Log import LogOutput

results_path = './results/17_02_19_UNMERGED'

Log.output = LogOutput.BOTH
Log.path = results_path
Log.clear()

base_path = "/home/stefano/Documents/University/DissertationDatasets"
pan12_dir = f"{base_path}/pan12-sexual-predator-identification-test-corpus-2012-05-21"

parser = CorpusParser.factory(CorpusName.PAN12, merge_messages=False)
parser.set_source_directory(pan12_dir)
dataset = Dataset(parser.get_params(), CorpusName.PAN12)

#
#   Parse corpus into the dataset.
#
if parser.is_serialized():
    parser = parser.deserialize()
else:
    parser.parse()
    parser.serialize()

parser.log_info()

if dataset.is_serialized():
    dataset = dataset.deserialize()
    dataset.log_info()
else:
    dataset = parser.get_dataset()
    dataset.finalize()
    dataset.log_info()
    dataset.balance_negatives()
    dataset.log_info()
    dataset.serialize()

parser.dump(f"{results_path}/parsed_files")

#
#   Initialize the benchmark object with the dataset and
#   select the classifiers.
#
benchmark = Benchmark(dataset)
benchmark.add_classifier(ClassifierType.RandomForest)
benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
benchmark.add_classifier(ClassifierType.SupportVectorMachine)
benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)
benchmark.add_classifier(ClassifierType.LogisticRegression)
benchmark.initialize_classifiers()

#
#   Select the metrics to create the plots for.
#
benchmark.select_metrics(
    MetricType.ACCURACY,
    MetricType.PRECISION,
    MetricType.RECALL,
    MetricType.ROC
)

#
#   Run benchmark and plot the results.
#
benchmark.run(10)
benchmark.get_info()
benchmark.save_metrics(results_path)
benchmark.plot_metrics()
benchmark.clustering()
