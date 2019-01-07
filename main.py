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
from Utils import DataConverter, Log

pan12_dir = "/home/stefano/Downloads/pan12-sexual-predator-identification-test-corpus-2012-05-21"

parser = CorpusParser.factory(CorpusName.PAN12)
parser.set_source_directory(pan12_dir)

if parser.is_serialized():
    parser = parser.deserialize()
else:
    parser.parse()
    parser.serialize()

conv = len(parser.conversations)
tot = 0
flagged = 0

Log.info("start")

auths = set()


for conversation in parser.conversations:
    tot += len(conversation.messages)
    if conversation.is_suspicious():
        flagged += 1
        for auth in conversation.authors:
            if auth.is_suspect():
                auths.add(auth)

print(f"Total conversations: {conv}")
print(f"Flagged conversations: {flagged}")
print(f"Avg messages per conv: {tot/conv}")
print(f"Total suspicious messages: {parser.get_perverted_messages_no()}")
print(f"Suspicious authors: {parser.get_perverted_authors_no()}")

Log.info("end")
# Log.info("Starting dataframe creation")
#
# conversations = parser.conversations
# all_messages = pd.DataFrame()
#
# for conversation in conversations:
#     messages = conversation.to_dictionary_list()
#     df = DataConverter.dictionary_list_to_dataframe(messages)
#     all_messages = all_messages.append(df)
#
# Log.info("Finished creating dataframe")


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
