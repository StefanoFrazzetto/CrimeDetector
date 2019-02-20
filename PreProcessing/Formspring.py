from enum import Enum

import pandas as pd

from Data import Dataset
from PreProcessing import CorpusParser


class User(object):
    def __init__(self, user_id: str):
        self.user_id = user_id


class PostLabel(Enum):
    NEGATIVE = 0
    POSITIVE = 1


class Post(object):
    def __init__(self):
        self.user = None
        self.asker = None

        self.post = ""
        self.question = ""
        self.answer = ""


class FormspringParser(CorpusParser):
    raw: pd.DataFrame

    def __init__(self):
        super(FormspringParser, self).__init__()
        self.raw = None

    def parse(self):
        self.raw = pd.read_csv(
            filepath_or_buffer=self.source_path,
            sep='\t'
        )

    def dump(self, directory, *args):
        pass

    def add_to_dataset(self, dataset: Dataset):
        pass

    def log_info(self):
        pass

    def _get_bully3(self):
        return self.raw.query('not bully3.isnull() and bully3 != "None"')
