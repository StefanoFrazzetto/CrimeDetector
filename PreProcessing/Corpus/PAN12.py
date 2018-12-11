from enum import Enum
from typing import List, Set

import pandas as pd

from Utils import Assert


class AuthorLabel(Enum):
    NORMAL = 0
    SUSPECT = 1


class MessageLabel(Enum):
    NORMAL = 0
    SUSPICIOUS = 1


class ConversationLabel(Enum):
    NORMAL = 0
    SUSPICIOUS = 1


class Author(object):
    id: str
    label: AuthorLabel

    def __init__(self, author_id: str):
        self.id = author_id
        self.label = AuthorLabel.NORMAL

    def __eq__(self, other: 'Author'):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def flag(self):
        self.label = AuthorLabel.SUSPECT

    def is_suspect(self):
        return self.label == AuthorLabel.SUSPECT


class Message(object):
    def __init__(self, author: Author, message_id: int, time: str, text: str,
                 label: MessageLabel = MessageLabel.NORMAL):
        self.author = author
        self.id = [int(message_id)]
        self.time = time
        self.text = text
        self.label = label

    def flag(self):
        self.label = MessageLabel.SUSPICIOUS

    def is_suspicious(self):
        return self.label == MessageLabel.SUSPICIOUS

    def join(self, message: 'Message'):
        """
        Concatenate the messages assuming that the parameter is older than self.
        :param message:
        :return:
        """
        # Check same author
        Assert.equal(self.author, message.author, "Cannot join messages from two different authors.")
        # Check different messages
        # assert len(set(self.id) - set(message.id)) > 0, "The messages may contain duplicates!"

        self.id = self.id + message.id
        self.time = message.time
        self.text = f"{self.text} {message.text}"

        self.label = MessageLabel.NORMAL if (
                self.label == MessageLabel.NORMAL and message.label == MessageLabel.NORMAL
        ) else MessageLabel.SUSPICIOUS

    def to_dictionary(self):
        return {'label': self.label.value, 'text': self.text}


class Conversation(object):
    id: str
    authors: Set[Author]
    messages: List[Message]

    def __init__(self, conversation_id: str):
        self.id = conversation_id
        self.authors = set()
        self.label = ConversationLabel.NORMAL
        self.messages = []

    def add_message(self, message: Message):
        self.authors.add(message.author)
        self.messages.append(message)

    def get_author_messages(self, author: Author):
        for message in self.messages:
            if message.author == author:
                yield message

    def flag(self):
        self.label = ConversationLabel.SUSPICIOUS

    def is_suspicious(self):
        return self.label == ConversationLabel.SUSPICIOUS

    def to_dictionary_list(self) -> list:
        messages = []
        for message in self.messages:
            messages.append(message.to_dictionary())
        return messages
