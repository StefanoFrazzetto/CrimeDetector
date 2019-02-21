from enum import Enum
from typing import List

import pandas as pd

from Data import Dataset
from Interfaces import Analyzable, AnalyzableLabel
from PreProcessing import CorpusParser
from Utils import Assert, Log


class User(object):
    def __init__(self, user_id: str):
        self.user_id = user_id


class PostLabel(Enum):
    NEGATIVE = 0
    POSITIVE = 1


class Post(Analyzable):
    def __init__(self):
        self.user = None
        self.asker = None

        self.post = ""
        self.label = PostLabel.NEGATIVE
        self.question = ""
        self.answer = ""

    def flag(self):
        self.label = PostLabel.POSITIVE

    def get_label(self) -> AnalyzableLabel:
        return AnalyzableLabel.POSITIVE if self.label == PostLabel.POSITIVE else AnalyzableLabel.NEGATIVE

    def get_data(self):
        return f"{self.question} {self.answer}"


class FormspringParser(CorpusParser):
    raw: pd.DataFrame
    negative: List[Post]
    positive: List[Post]

    def __init__(self):
        super(FormspringParser, self).__init__()
        self.raw = None

        self.negative = []
        self.positive = []

    def add_to_dataset(self, dataset: Dataset):
        elements = self.negative + self.positive
        for element in elements:
            dataset.put(element)

    def dump(self, directory, *args):
        raise NotImplementedError()

    def log_info(self):
        Log.info(f"Positive: {len(self.positive)} - Negative: {len(self.negative)} / Total: {len(self.raw)}")

    def parse(self):
        self.raw = pd.read_csv(
            filepath_or_buffer=self.source_path,
            sep='\t'
        )

        # NEGATIVE
        for _, element in self._get_non_bully().iterrows():
            post = self._create_post(element)
            self.negative.append(post)

        # POSITIVE
        for _, element in self._get_bully().iterrows():
            post = self._create_post(element)
            post.flag()
            self.positive.append(post)

        # Check that all elements have been added
        self._do_sanity_check()

    @staticmethod
    def _create_post(element: pd.Series) -> Post:
        post = Post()
        post.user = element['userid']
        post.post = element['post']
        post.question = element['ques']
        post.answer = element['ans']
        post.asker = element['asker']
        return post

    def _do_sanity_check(self):
        parsed = self.negative + self.positive
        Assert.same_length(parsed, self.raw)

    def _get_bully(self):
        query = '(not bully1.isnull() and bully1 != "None") or ' \
                '(not bully2.isnull() and bully2 != "None") or ' \
                '(not bully3.isnull() and bully3 != "None")'
        return self.raw.query(query)

    def _get_non_bully(self):
        query = '(bully1.isnull() or bully1 == "None") and ' \
                '(bully2.isnull() or bully2 == "None") and ' \
                '(bully3.isnull() or bully3 == "None")'
        return self.raw.query(query)
