import html
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
    """
    Specialized parser for the Formspring v4 dataset in csv format.
    """

    raw: pd.DataFrame
    negative: List[Post]
    positive: List[Post]

    def __init__(self, **kwargs):
        super(FormspringParser, self).__init__()
        self.raw = None
        self.democratic = kwargs.pop('democratic', False)
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

    def _do_parse(self):
        self.raw = pd.read_csv(
            filepath_or_buffer=self.source_path,
            sep='\t',
            engine='python'
        )

        # Fill NaN column values with empty strings.
        self.raw.post.fillna('', inplace=True)
        self.raw.ques.fillna('', inplace=True)
        self.raw.ans.fillna('', inplace=True)

        positive = self._get_bully()
        negative = pd.concat([self.raw, positive, positive]).drop_duplicates(keep=False)

        # NEGATIVE
        for _, element in negative.iterrows():
            post = self._create_post(element)
            self.negative.append(post)

        # POSITIVE
        for _, element in positive.iterrows():
            post = self._create_post(element)
            post.flag()
            self.positive.append(post)

        # Check that all elements have been added
        self._do_sanity_check()

    @staticmethod
    def _create_post(element: pd.Series) -> Post:
        post = Post()
        post.user = element['userid']
        post.post = html.unescape(element['post'])
        post.question = html.unescape(element['ques'])
        post.answer = html.unescape(element['ans'])
        post.asker = element['asker']
        return post

    def _do_sanity_check(self):
        parsed = self.negative + self.positive
        Assert.same_length(parsed, self.raw)

    def _get_bully(self):
        # Require at least two people to consider the entry as cyberbullying
        if self.democratic:
            first = '(ans1 == "Yes") and (severity1 != "0")'
            second = '(ans2 == "Yes") and (severity2 != "0")'
            third = '(ans3 == "Yes") and (severity3 != "0")'
            query = f"({first} and {second}) or ({second} and {third}) or ({third} and {first})"

        # Require at least one person to consider the entry as cyberbullying
        else:
            query = '(not bully1.isnull() and bully1 != "None") or ' \
                    '(not bully2.isnull() and bully2 != "None") or ' \
                    '(not bully3.isnull() and bully3 != "None")'

        return self.raw.query(query)
