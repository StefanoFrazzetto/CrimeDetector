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
        return {'label': self.label.value, 'text': self.text}


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


class Parser(CorpusParser):
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

    def __init__(self, merge_messages: bool = True):
        super(Parser, self).__init__(merge_messages=merge_messages)

        self.problem1 = []
        self.problem2 = defaultdict(list)
        self.conversations = []

    def parse(self):
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

        Log.info("### PARSER INFO ###", header=True)
        Log.info(f"Flagged conversations (F): {flagged_conversations} / "
                 f"Total conversations: {len(self.conversations)} - "
                 f"Ratio (F/T): {Numbers.get_formatted_percentage(flagged_conversations, len(self.conversations))} %")

        Log.info(f"Flagged messages (F): {flagged_messages} / "
                 f"Total messages (T): {total_messages} - "
                 f"Ratio (F/T): {Numbers.get_formatted_percentage(flagged_messages, total_messages)} %")

        Log.info(f"Flagged authors (F): {flagged_authors} / "
                 f"Total authors (T): {len(authors)} - "
                 f"Ratio (F/T): {Numbers.get_formatted_percentage(flagged_authors, len(authors))} %")

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
        Log.info(f"Parsing {self.corpus_name.name} corpus... ", newline=False)

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

                # If 'merge_messages' is false, just add the current message and continue.
                if not self.merge_messages:
                    conversation.add_message(current_message)
                    continue

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
            # This occurs only when merging messages.
            if not message_added and self.merge_messages:
                conversation.add_message(previous_message)

            self.conversations.append(conversation)
        Log.info("done.", timestamp=False)

    def dump(self, directory, *args):
        for conversation in self.conversations:
            if conversation.is_suspicious():  # SUSP only
                dir_name = f"{directory}/{conversation.id}"
                # File.create_directory(dir_name)
                for message in conversation.messages:
                    file_name = f"{directory}/{conversation.id}.txt"
                    if message.is_suspicious():
                        file_content = f"[FLAGGED]{message.id}: {message.text}"
                    else:
                        file_content = f"{message.id}: {message.text}"
                    File.write_file(file_name, file_content, mode="a+")
                    File.write_file(file_name, "\n\n----------------------------------\n\n", mode="a+")

    def get_dataset(self) -> Dataset:
        dataset = Dataset(self.corpus_name)
        for conversation in self.conversations:
            for message in conversation.messages:
                dataset.put(message)

        return dataset
