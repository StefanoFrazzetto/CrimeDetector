"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""

from Classification import Benchmark, ClassifierType, MetricType, FeatureExtraction
from Classification.FeatureExtraction import FeatureExtractionStep
from Data import Dataset
from PreProcessing import CorpusName, CorpusParser
from Utils import Log
from Utils.Log import LogOutput, LogLevel

#
#   Base variables.
#
results_path = './results/18_02_19-1'
base_path = "/home/stefano/Documents/University/DissertationDatasets"
pan12_dir = f"{base_path}/pan12-sexual-predator-identification-test-corpus-2012-05-21"

#
#   Logging options.
#
Log.level = LogLevel.INFO
Log.output = LogOutput.BOTH
Log.path = results_path
Log.clear()

Log.info("===============================================", header=True, timestamp=False)
Log.info("===========     PROCESS STARTED     ===========", header=True, timestamp=False)
Log.info("===============================================", header=True, timestamp=False)

parser = CorpusParser.factory(CorpusName.PAN12, merge_messages=False)
parser.set_source_directory(pan12_dir)
dataset = Dataset(parser.get_params(), CorpusName.PAN12)

#
#   Parse corpus into the dataset.
#
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
    # parser.dump(f"{results_path}/parsed_files")

    dataset.finalize()
    dataset.log_info()
    dataset.balance_negatives()
    dataset.log_info()
    dataset.serialize()

#
#   Initialize FeatureExtraction pipeline.
#
feature_extraction = FeatureExtraction(
    FeatureExtractionStep.VECTORIZE,
    # FeatureExtractionStep.TOKENIZE,
    FeatureExtractionStep.TFIDF,
    dataset=dataset,
    max_features=None,
)

#
#   Initialize the benchmark object with the dataset and
#   select the classifiers.
#
benchmark = Benchmark(dataset=dataset, feature_extraction=feature_extraction)
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
    MetricType.F05,
    MetricType.F1,
    MetricType.F3,
    MetricType.ROC,
)

#
#   Run benchmark and plot the results.
#
benchmark.run(10)
benchmark.get_info()
benchmark.save_metrics(results_path)
# benchmark.plot_metrics()
# benchmark.clustering()

Log.info("==========      PROCESS FINISHED      ==========", header=True, timestamp=False)
