"""
Useful links:
- https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475
- https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73
"""

# from Classification.Classifiers import ClassifierType, Classifier
#
# classifier = Classifier.factory(
#     classifier_type=ClassifierType.NAIVE_BAYES,
#     split_ratio=0.7,
#     corpus_directory='/home/stefano/Downloads/maildir',
#     load_percentage=1
# )
#
# classifier.get_accuracy()
from Classification.Data import Dataset, DataCategory
from PreProcessing.DataReader import DataReader

dataset = Dataset()
dr = DataReader(dataset)
dr.add_exclusion("Acknowldegement")
dir = "/home/stefano/Downloads/spam-non-spam-dataset"
dr.add_dir_to_dataset(f"{dir}/train-mails", DataCategory.TRAINING)
dr.add_dir_to_dataset(f"{dir}/test-mails", DataCategory.TRAINING)

hamc, spamc = dataset.get_ham_spam_count()

print(f"Ham count: {hamc} - Spam count: {spamc}")