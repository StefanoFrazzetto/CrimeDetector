from collections import defaultdict
from enum import Enum
from typing import List, Set
from xml.etree import cElementTree

from Data import Dataset
from Interfaces import Analyzable, AnalyzableLabel
from PreProcessing import CorpusParser
from Utils import Assert, Log, File, Numbers


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
    """
    Represents the author of a message.
    """
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


class Message(Analyzable):
    """
    Represents a message in a conversation.
    """
    author: Author

    id: List[int]
    time: str
    text: str
    label: MessageLabel

    def get_label(self) -> AnalyzableLabel:
        return AnalyzableLabel.NEGATIVE \
            if self.label == MessageLabel.NORMAL \
            else AnalyzableLabel.POSITIVE

    def get_data(self):
        return self.text

    def __init__(self, author: Author, message_id: int, time: str, text: str,
                 label: MessageLabel = MessageLabel.NORMAL):
        self.author = author
        self.id = [int(message_id)]
        self.time = time
        self.text = text
        self.label = label

    def __eq__(self, other):
        return self.id == other.id

    def flag(self):
        self.label = MessageLabel.SUSPICIOUS

    def is_suspicious(self):
        return self.label == MessageLabel.SUSPICIOUS

    def get_id(self):
        """
        Return the ID of the message.
        If the message was joined with other messages, it will return the ID
        of the latest added message.
        :return:
        """
        return self.id[-1]

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
        return {'label': self.label.value, 'data': self.text}


class Conversation(object):
    """
    Represents a conversation between a number of authors.
    """
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


class PAN12Parser(CorpusParser):
    problem1: List
    problem2: defaultdict
    conversations: List[Conversation]

    """
    Specialized parser for the PAN-12 dataset.
    
    Problem 1 contains 254 ids of authors considered perverted (one per line).

    Problem 2 contains 6478 conversations id and lines id of those line considered suspicious
    (of a perverted behavior) in a particular conversation.
    """

    xml_file = "pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"

    problem1_file = "pan12-sexual-predator-identification-groundtruth-problem1.txt"
    problem2_file = "pan12-sexual-predator-identification-groundtruth-problem2.txt"

    def __init__(self, **kwargs):
        super(PAN12Parser, self).__init__()

        self.merge_messages = kwargs.pop('merge_messages', False)
        self.problem1 = []
        self.problem2 = defaultdict(list)
        self.conversations = []

    def _do_parse(self):
        self.__load_problems(self.problem1_file, self.problem2_file)
        self.__parse_xml(self.xml_file)

    def log_info(self):
        authors = set()
        flagged_authors = 0
        flagged_conversations = 0
        flagged_messages = 0
        total_messages = 0

        for conversation in self.conversations:
            for author in conversation.authors:
                authors.add(author)

            if conversation.is_suspicious():
                flagged_conversations += 1

            for message in conversation.messages:
                total_messages += 1
                if message.is_suspicious():
                    flagged_messages += 1

        for author in authors:
            if author.is_suspect():
                flagged_authors += 1

        # Table header
        data = [["Element", "Total", "No. Flagged", "Flagged %"]]

        conversations = [
            "Conversations",
            len(self.conversations),
            flagged_conversations,
            Numbers.percentage(flagged_conversations, len(self.conversations))
        ]

        messages = [
            "Messages",
            total_messages,
            flagged_messages,
            Numbers.percentage(flagged_messages, total_messages)
        ]

        authors = [
            "Authors",
            len(authors),
            flagged_authors,
            Numbers.percentage(flagged_authors, len(authors))
        ]

        # Append lists to table
        data.extend([conversations, messages, authors])

        Log.tabulate(data, floatfmt=(".0f", ".0f", ".0f", ".2f"))

    def __load_problems(self, problem1_file, problem2_file):
        Log.info("Loading ground truth files... ", newline=False)

        # Problem 1
        with open(f"{self.source_path}/{problem1_file}") as f:
            for line in f:
                self.problem1.append(line.rstrip())

        # Problem 2
        with open(f"{self.source_path}/{problem2_file}") as f:
            for current_line in f:
                line = current_line.split()
                conversation_id = line[0]  # the conversation id
                line_no = int(line[1])  # the id (id) of the incriminated message
                self.problem2[conversation_id].append(line_no)

        Log.info("done.", timestamp=False)

        # Check that the length of the parsed content is correct.
        problem1_parsed_lines = self.get_perverted_authors_no()
        problem1_expected_lines = File.length(f"{self.source_path}/{problem1_file}")

        problem2_parsed_lines = self.get_perverted_messages_no()
        problem2_expected_lines = File.length(f"{self.source_path}/{problem2_file}")

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

        # Parse the XML document and get its root node
        document = cElementTree.parse(f"{self.source_path}/{xml_file}")
        document_root = document.getroot()

        # Loop through <conversation> nodes
        for current_conversation in document_root.iter('conversation'):

            conversation = Conversation(current_conversation.get('id'))

            # Loops through the messages in the current conversation
            for current_message in current_conversation.iter('message'):

                current_author = Author(current_message.find('author').text)
                current_message_line = current_message.get('line')
                current_message_time = current_message.find('time').text
                current_message_text = current_message.find('text').text

                # Flag author if their ID is in problem1 file
                if current_author.id in self.problem1:
                    current_author.flag()

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
                if conversation.id in self.problem2:
                    if current_message.get_id() in self.problem2.get(conversation.id):
                        # Yes, flag message, author, and conversation.
                        current_message.flag()
                        current_author.flag()
                        conversation.flag()

                conversation.add_message(current_message)

            self.conversations.append(conversation)

        if self.merge_messages:
            self.do_merge_messages()

    def do_merge_messages(self):
        """
        Merge messages in a conversation if they belong to the same author and they are consecutive.
        """

        Log.info("Merging messages...")
        for conversation in self.conversations:
            for i in range(len(conversation.messages) - 1, -1, -1):
                recent_message = conversation.messages[i]
                older_message = conversation.messages[i - 1]

                if recent_message.author == older_message.author:
                    older_message.join(recent_message)
                    conversation.messages.remove(recent_message)

    def dump(self, directory, *args):
        if not File.directory_exists(directory):
            File.create_directory(directory)
        else:
            File.empty_directory(directory)

        for conversation in self.conversations:
            if conversation.is_suspicious():  # SUSP only
                file_name = f"{directory}/{conversation.id}.txt"
                for message in conversation.messages:
                    if message.author.is_suspect():
                        file_content = f"[SUSPECT]{message.id}: {message.text}"
                    else:
                        file_content = f"[USER]{message.id}: {message.text}"

                    if message.is_suspicious():
                        file_content = "[FLAGGED]" + file_content

                    File.write_file(file_name, file_content, mode="a+")
                    File.write_file(file_name, "\n\n----------------------------------\n\n", mode="a+")

    def add_to_dataset(self, dataset: Dataset):
        for conversation in self.conversations:
            for message in conversation.messages:
                dataset.put(message)
