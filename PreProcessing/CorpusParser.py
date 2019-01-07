import abc
import hashlib
from collections import defaultdict
from xml.etree import cElementTree

from Interfaces import Serializable
from PreProcessing.Corpus.PAN12 import *
from Utils import Log, File


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

    # xml_training_file = "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
    # problem1_training_file = "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
    # problem2_training_file = "pan12-sexual-predator-identification-diff.txt"

    def __init__(self):
        super(PAN12Parser, self).__init__(CorpusType.SEXUAL_PREDATORS)

        self.problem1 = []
        self.problem2 = defaultdict(list)
        self.conversations = []

    def __load_problems(self, problem1_file, problem2_file):
        Log.info("Loading ground truth files... ", newline=False)

        # Problem 1
        with open(f"{self.source_directory}/{problem1_file}") as f:
            for line in f:
                self.problem1.append(line.rstrip())

        # Problem 2
        with open(f"{self.source_directory}/{problem2_file}") as f:
            for current_line in f:
                line = current_line.split()
                conversation_id = line[0]  # the conversation id
                line_no = int(line[1])  # the id (id) of the incriminated message
                self.problem2[conversation_id].append(line_no)

        Log.info("done.", timestamp=False)

        # Check that the length of the parsed content is correct.
        problem1_parsed_lines = self.get_perverted_authors_no()
        problem1_expected_lines = File.length(f"{self.source_directory}/{problem1_file}")

        problem2_parsed_lines = self.get_perverted_messages_no()
        problem2_expected_lines = File.length(f"{self.source_directory}/{problem2_file}")

        Assert.equal(problem1_parsed_lines, problem1_expected_lines, "Check problem 1 parsing.")
        Assert.equal(problem2_parsed_lines, problem2_expected_lines, "Check problem 2 parsing.")

        Log.info("Problem files parsed successfully.")

    def get_perverted_messages_no(self):
        problem2_parsed_lines = 0
        for _, value in self.problem2.items():
            problem2_parsed_lines += len(value)
        return problem2_parsed_lines

    def get_perverted_authors_no(self):
        return len(self.problem1)

    def __parse_xml(self, xml_file):
        """
        Parse the XML file into the internal object representation.
        :return: List[Conversation]
        """
        Log.info(f"Parsing {self.corpus_type} corpus...", newline=False)

        # Parse the XML document and get its root node
        document = cElementTree.parse(f"{self.source_directory}/{xml_file}")
        document_root = document.getroot()

        # Loop through <conversation> nodes
        for current_conversation in document_root.iter('conversation'):

            conversation = Conversation(current_conversation.get('id'))

            # Initialize vars
            previous_author = Author("RANDOM_AUTHOR_ID")
            previous_message = None
            message_added = False

            # Loops through the messages in the current conversation
            for current_message in current_conversation.iter('message'):

                current_author = Author(current_message.find('author').text)
                current_message_line = current_message.get('line')
                current_message_time = current_message.find('time').text
                current_message_text = current_message.find('text').text

                # Occurs with empty text tags, e.g. <text />
                if current_message_text is None:
                    current_message_text = ""

                current_message = Message(
                    current_author,
                    current_message_line,
                    current_message_time,
                    current_message_text
                )

                # Check if the message is marked in the problem file
                if self.problem2.get(conversation.id) is not None:
                    if current_message.get_id() in self.problem2.get(conversation.id):
                        # Yes, flag message, author, and conversation.
                        current_message.flag()
                        current_message.author.flag()
                        conversation.flag()

                # If the author is the same of the previous message, merge the messages.
                # This is done to create a more complete structure of the messages, and
                # also to save space.
                if current_author == previous_author:
                    previous_message.join(current_message)
                    message_added = False

                # Otherwise, if a different author, add the previous message to the
                # current conversation and set the current message as 'previous'.
                else:
                    # The variable is set to 'None' before the first iteration.
                    if previous_message is not None:
                        conversation.add_message(previous_message)
                        message_added = True
                    previous_author = current_author
                    previous_message = current_message

            # Add the last message of the conversation.
            if not message_added:
                conversation.add_message(previous_message)

            self.conversations.append(conversation)
        Log.info("done.", timestamp=False)

    def parse(self):
        self.__load_problems(self.problem1_file, self.problem2_file)
        self.__parse_xml(self.xml_file)
        # self.__join_messages()
