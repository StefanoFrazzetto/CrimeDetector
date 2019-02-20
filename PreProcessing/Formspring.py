from Data import Dataset
from PreProcessing import CorpusParser


class FormspringParser(CorpusParser):

    def __init__(self):
        super(FormspringParser, self).__init__()

    def parse(self):
        pass

    def dump(self, directory, *args):
        pass

    def add_to_dataset(self, dataset: Dataset):
        pass

    def log_info(self):
        pass
