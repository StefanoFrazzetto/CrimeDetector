import abc
import hashlib
from collections import defaultdict
from xml.etree import cElementTree

from Interfaces import Serializable
from PreProcessing.Corpus.PAN12 import *
from Utils import Log


class CorpusType(Enum):
    SEXUAL_PREDATORS = 0
    CYBER_BULLYING = 1
    BANKING_FRAUD = 2

    def __str__(self):
        return self.name


class CorpusName(Enum):
    PAN12 = 0


class CorpusParser(Serializable, metaclass=abc.ABCMeta):

    def __init__(self, corpus_type: CorpusType):
        self.corpus_type = corpus_type
        self.source_directory = ""

    def __eq__(self, other: 'CorpusParser'):
        return self.source_directory == other.source_directory

    def __hash__(self):
        obj_hash = str(self.source_directory).encode('utf-8')
        return hashlib.sha256(obj_hash).hexdigest()

    @staticmethod
    def factory(corpus_name: CorpusName):
        if corpus_name == CorpusName.PAN12:
            return PAN12Parser()

        assert f"Unknown corpus labeler {corpus_name}"

    def set_source_directory(self, source_directory: str):
        self.source_directory = source_directory

    @abc.abstractmethod
    def parse(self):
        pass


class PAN12Parser(CorpusParser):
    problem1: List
    problem2: defaultdict
    conversations: List[Conversation]

    """
    Problem 1 contains 254 ids of authors considered perverted (one per line).

    Problem 2 contains 6478 conversations id and lines id of those line considered suspicious
    (of a perverted behavior) in a particular conversation.
    """

    xml_file = "pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"
    problem1_file = "pan12-sexual-predator-identification-groundtruth-problem1.txt"
    problem2_file = "pan12-sexual-predator-identification-groundtruth-problem2.txt"

    xml_training_file = "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
    problem1_training_file = "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
    problem2_training_file = "pan12-sexual-predator-identification-diff.txt"

    def __init__(self):
        super(PAN12Parser, self).__init__(CorpusType.SEXUAL_PREDATORS)

        self.problem1 = []
        self.problem2 = defaultdict(list)
        self.conversations = []

    def __load_problems(self):
        Log.info("Loading problems... ", newline=False)

        # Problem 1
        with open(f"{self.source_directory}/{self.problem1_training_file}") as f:
            for line in f:
                self.problem1.append(line)

        # Problem 2
        with open(f"{self.source_directory}/{self.problem2_training_file}") as f:
            for current_line in f:
                line = current_line.split()
                conversation_id = line[0]  # the conversation id
                line_no = int(line[1])  # the id (id) of the incriminated message
                self.problem2[conversation_id].append(line_no)

        Log.info("done.", timestamp=False)

    def __parse_xml(self):
        """
        Parse the XML file into the internal object representation.
        :return: List[Conversation]
        """
        Log.info(f"Parsing {self.corpus_type} corpus...")

        document = cElementTree.parse(f"{self.source_directory}/{self.xml_training_file}")
        document_root = document.getroot()

        # Loop through <conversation>
        for current_conversation in document_root.iter('conversation'):

            conversation = Conversation(current_conversation.get('id'))
            previous_author = Author("RANDOM_AUTHOR_ID")
            previous_message = None
            message_added = False

            for current_message in current_conversation.iter('message'):

                current_author = Author(current_message.find('author').text)
                current_message = Message(
                    current_author,
                    current_message.get('line'),
                    current_message.find('time').text,
                    current_message.find('text').text
                )

                # Check if the message is suspicious
                if self.problem2.get(conversation.id) is not None:
                    if current_message.get_id() in self.problem2.get(conversation.id):
                        # Yes, flag message, author, and conversation.
                        current_message.flag()
                        current_message.author.flag()
                        conversation.flag()

                if current_author == previous_author:
                    previous_message.join(current_message)
                    message_added = False
                else:
                    if previous_message is not None:
                        conversation.add_message(previous_message)
                        message_added = True
                    previous_author = current_author
                    previous_message = current_message

            # Handle specific case (author of last messages)
            if not message_added:
                conversation.add_message(previous_message)

            self.conversations.append(conversation)
        Log.info("done.")

    # def __join_messages(self):
    #     for conversation in self.conversations:
    #         for i in range(len(conversation.messages) - 1):
    #             # Same author
    #             if conversation.messages[i].author == conversation.messages[i + 1].author:
    #                 conversation.messages[i] = conversation.messages[i].join(conversation.messages[i + 1])
    #             else:
    #                 # Was already joined with the previous ???
    #                 del conversation.messages[i]

    def parse(self):
        self.__load_problems()
        self.__parse_xml()
        # self.__join_messages()
