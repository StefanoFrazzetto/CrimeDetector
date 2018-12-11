"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""
import pandas as pd

from PreProcessing import CorpusName, CorpusParser

# CREATE DATASET
from PreProcessing.Corpus.PAN12 import Conversation
from Utils import DataConverter

pan12_dir = "/home/stefano/Documents/University/DissertationDatasets/pan12-sexual-predator-identification-test-corpus-2012-05-21"

parser = CorpusParser.factory(CorpusName.PAN12)
parser.set_source_directory(pan12_dir)

if parser.is_serialized():
    parser = parser.deserialize()
else:
    parser.parse()
    parser.serialize()

conversations = parser.conversations
all_messages = pd.DataFrame()

for conversation in conversations:
    messages = conversation.to_dictionary_list()
    df = DataConverter.dictionary_list_to_dataframe(messages)
    all_messages = all_messages.append(df)

print("ciao")


# dataset = Dataset.deserialize() if Dataset.is_serialized() else Dataset()
# if dataset.is_serialized() is False:
#     # Create the dataset and serialize it.
#     data_reader = CorpusReader(CorpusName.PAN12, dataset)
#     # base_dir = "/home/stefano/Documents/University/DissertationDatasets/pan12_parsed"
#     base_dir = "/home/stefano/Downloads/spam-non-spam-dataset"
#     data_reader.add_exclusion("Acknowldegement")
#
#     data_reader.add_dir_to_dataset(f"{base_dir}/test-mails")
#     data_reader.add_dir_to_dataset(f"{base_dir}/train-mails")
#
#     # data_reader.add_dir_to_dataset(f"{base_dir}/positive", None, DataLabel.POSITIVE)
#     # dataset.print_info()
#     #
#     # data_reader.add_dir_to_dataset(f"{base_dir}/negative", None, DataLabel.NEGATIVE, 0.1)
#
#     dataset.serialize()
#
# dataset.print_info()
#
# benchmark = Benchmark(dataset)
# benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
# benchmark.add_classifier(ClassifierType.SupportVectorMachine)
# benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)
# benchmark.run()
