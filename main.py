"""
Author:
Stefano Frazzetto
BSc (Hons) Applied Computing

Faculty of Natural Sciences
Department of Computing Science and Mathematics
University of Stirling
"""
import os
import sys

from Classification import Benchmark
from Data import Dataset
from PreProcessing import CorpusName, CorpusParser
from Utils import Log

base_path = "/home/stefano/Documents/University/DissertationDatasets"
pan12_dir = f"{base_path}/pan12-sexual-predator-identification-test-corpus-2012-05-21"

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
print(f"Avg messages per conv: {tot / conv}")
print(f"Total suspicious messages: {parser.get_perverted_messages_no()}")
print(f"Suspicious authors: {parser.get_perverted_authors_no()}")

Log.info("end")

parser.log_info()

sys.exit(1)

dataset = Dataset()
if dataset.is_serialized():
    dataset = dataset.deserialize()
else:
    dataset = parser.get_dataset()
    dataset.serialize()
    dataset.log_info()

benchmark = Benchmark(dataset)
# benchmark.add_classifier(ClassifierType.MultiLayerPerceptron)
# benchmark.add_classifier(ClassifierType.SupportVectorMachine)
# benchmark.add_classifier(ClassifierType.MultinomialNaiveBayes)
# benchmark.run()
