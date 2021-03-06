import os
from typing import List

from Classification import Benchmark, ClassifierType, MetricType, FeatureExtraction, FeatureExtractionStep
from Data import Dataset
from PreProcessing import CorpusName, CorpusParser
from Utils import Log, Time, File
from Utils.Log import LogOutput, LogLevel


class Corpus(object):
    """
    Base corpus information.
    """

    name: CorpusName
    path: str
    parsing_options: List[dict]

    def __init__(self, name: CorpusName, path: str, parsing_options: list):
        self.name = name
        self.path = path
        self.parsing_options = parsing_options


class Scenario(object):

    """
    Represent a scenario, as described in the dissertation report.
    """

    def __init__(self, name: str, *feature_extraction_steps: FeatureExtractionStep):
        self.name = name
        self.feature_extraction_steps = feature_extraction_steps
        self.results_path = self._get_results_path()

    def _get_results_path(self):
        return os.path.abspath(f'./results/{Time.get_timestamp("%Y-%m-%d")}/Scenarios/{self.name}')

    @staticmethod
    def _get_dataset(parser: CorpusParser, dataset: Dataset):
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

        return dataset

    def run(self, run_corpus: Corpus = None):
        for corpus in CORPORA:

            # Selectively run corpora
            if run_corpus is None or run_corpus == corpus:
                corpus_results_path = f"{self.results_path}/{corpus.name.name}/"
                for parsing_options in corpus.parsing_options:
                    parser_config_str = f"{list(parsing_options.items())[0][0]}_{str(list(parsing_options.items())[0][1])}"
                    results_path = f"{corpus_results_path}/{parser_config_str}"

                    scenario_log = f"### SCENARIO {self.name} for {corpus.name.name} using {parser_config_str} ###"
                    if File.directory_exists(results_path):
                        print(f"SKIPPING: {scenario_log}")
                        continue

                    # Configure log options.
                    Log.level = LogLevel.INFO
                    Log.output = LogOutput.BOTH
                    Log.path = results_path
                    Log.init()

                    Log.info(scenario_log)

                    # Create parser and dataset.
                    parser = CorpusParser.factory(corpus_name=corpus.name, source_path=corpus.path, **parsing_options)
                    dataset = Dataset(parser.get_params(), corpus_name=corpus.name)
                    dataset = self._get_dataset(parser, dataset)

                    dataset.balance_all(5)

                    # Create feature extraction pipeline.
                    feature_extraction = FeatureExtraction(
                        *self.feature_extraction_steps,
                        dataset=dataset,
                        max_features=None
                    )

                    # Set benchmark options.
                    benchmark = Benchmark(dataset=dataset, feature_extraction=feature_extraction)
                    benchmark.add_classifier(ClassifierType.RandomForest)
                    benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
                    benchmark.add_classifier(ClassifierType.SupportVectorMachine)
                    benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)
                    benchmark.add_classifier(ClassifierType.LogisticRegression)
                    benchmark.initialize_classifiers()

                    # Select benchmark metrics.
                    benchmark.select_metrics(
                        MetricType.ACCURACY,
                        MetricType.PRECISION,
                        MetricType.RECALL,
                        MetricType.F1,
                        MetricType.ROC,
                        MetricType.MCC,
                        MetricType.CONFUSION_MATRIX
                    )

                    # Start process and save results.
                    benchmark.run(10)
                    benchmark.get_info()
                    benchmark.save_metrics(results_path)


class ScenarioRunner(object):
    SCENARIOS: List[Scenario] = [
        Scenario("A", FeatureExtractionStep.UNDERSAMPLE_DROP),
        Scenario("B", FeatureExtractionStep.OVERSAMPLE_ADASYN),

        Scenario("C", FeatureExtractionStep.TOKENIZE, FeatureExtractionStep.UNDERSAMPLE_DROP),
        Scenario("D", FeatureExtractionStep.TOKENIZE, FeatureExtractionStep.OVERSAMPLE_ADASYN),

        Scenario("E", FeatureExtractionStep.TFIDF, FeatureExtractionStep.UNDERSAMPLE_DROP),
        Scenario("F", FeatureExtractionStep.TFIDF, FeatureExtractionStep.OVERSAMPLE_ADASYN),

        Scenario("G",
                 FeatureExtractionStep.TOKENIZE,
                 FeatureExtractionStep.TFIDF,
                 FeatureExtractionStep.UNDERSAMPLE_DROP),
        Scenario("H",
                 FeatureExtractionStep.TOKENIZE,
                 FeatureExtractionStep.TFIDF,
                 FeatureExtractionStep.OVERSAMPLE_ADASYN),
    ]

    @staticmethod
    def run(name: str, corpus: Corpus = None):
        """
        Run a single scenario, optionally only for a single corpus.
        :param name:
        :param corpus:
        :return:
        """
        for scenario in ScenarioRunner.SCENARIOS:
            if scenario.name == name:
                scenario.run(corpus)

    @staticmethod
    def run_all():
        """
        Run all the scenarios, i.e. from A to H.
        :return:
        """
        for scenario in ScenarioRunner.SCENARIOS:
            scenario.run()


DATASETS_PATH = "./datasets"
PAN12_PATH = f"{DATASETS_PATH}/pan12"
FORMSPRING_FILE_PATH = f"{DATASETS_PATH}/formspring/formspring_data.csv"

CORPORA: List[Corpus] = [
    Corpus(
        CorpusName.PAN12,
        PAN12_PATH,
        [{"merge_messages": False}, {"merge_messages": True}]
    ),

    Corpus(
        CorpusName.FORMSPRING,
        FORMSPRING_FILE_PATH,
        [{"democratic": False}, {"democratic": True}]
    )
]

# Run a single scenario.
# ScenarioRunner.run("A", CORPORA[1])

# Run all scenarios.
ScenarioRunner.run_all()
