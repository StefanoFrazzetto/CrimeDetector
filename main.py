"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""

from Classification import Benchmark, ClassifierType, MetricType, FeatureExtraction, FeatureExtractionStep
from Data import Dataset
from PreProcessing import CorpusName, CorpusParser
from Utils import Log, Time
from Utils.Log import LogOutput, LogLevel

#
#   Base variables.
#
base_path = "./datasets"
pan12_dir = f"{base_path}/pan12"
formspring_file = f"{base_path}/formspring/formspring_data.csv"

corpus = CorpusName.PAN12
corpus_path = pan12_dir

results_path = f'./results/{Time.get_timestamp("%Y-%m-%d")}_{corpus.name}'

#
#   Logging options.
#
Log.level = LogLevel.INFO
Log.output = LogOutput.BOTH
Log.path = results_path
Log.clear()
Log.init()

Log.info("===============================================", header=True, timestamp=False)
Log.info("===========     PROCESS STARTED     ===========", header=True, timestamp=False)
Log.info("===============================================", header=True, timestamp=False)

# parser = CorpusParser.factory(CorpusName.FORMSPRING, formspring_file, merge_messages=False)
parser = CorpusParser.factory(corpus_name=corpus, source_path=corpus_path, merge_messages=True)
dataset = Dataset(parser.get_params(), corpus_name=corpus)

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

    dataset.finalize()
    dataset.log_info()
    dataset.serialize()

# Balance training and testing subsets with a 5:1 ratio, if achievable.
dataset.balance_all(5, random_state=None)

#
#   Initialize FeatureExtraction pipeline.
#
# noinspection PyUnreachableCode
feature_extraction = FeatureExtraction(
    FeatureExtractionStep.TOKENIZE,
    FeatureExtractionStep.TFIDF,
    # FeatureExtractionStep.OVERSAMPLE_ADASYN,
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
    MetricType.MCC,
    MetricType.CONFUSION_MATRIX
)

#
#   Run benchmark and plot the results.
#
benchmark.run(10)
benchmark.get_info()
benchmark.save_metrics(results_path)
# benchmark.plot_decision_function()
benchmark.clustering(draw_centroids=False, three_dimensional=False, save_path=results_path)
benchmark.clustering(draw_centroids=False, three_dimensional=True, save_path=results_path)

Log.info("==========      PROCESS FINISHED      ==========", header=True, timestamp=False)
