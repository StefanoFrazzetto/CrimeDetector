from math import floor

from Classification.Data import DataLabel
from Utils import File, Log


class LabelledEmailReader:
    """
    1. Get dir files (recursive)
    2. Read file
        2.1 Get body only
    3. Append data-class to dataframe
    """
    DEFAULT_ENCODING = "utf-8"

    @staticmethod
    def __get_body(file):
        text = File.read(file)
        is_body = False
        lines = []

        for line in text:
            if is_body:
                lines.append(line)
            elif line == "\n":
                is_body = True

        return '\n'.join(lines)


class PreProcessor:
    DEFAULT_LANGUAGE = "english"
    DEFAULT_LOAD_PERCENTAGE = 100
    DEFAULT_STOP_WORDS = True

    def __process_file(self, file):
        """Process a file and add it to the dataset."""
        Log.info(f"~ Processing {file}")
        text = File.read(file)
        # clean_text = TextUtils.clean(text)
        self.dataset.append(text)

    def __load_corpus(self):
        """Load the defined percentage of the corpus."""
        to_load = floor(self.corpus_size * (self.load_percentage / 100))
        to_process = floor(self.corpus_size * (self.load_percentage / 100))
        Log.info(f"# Loading ~{self.load_percentage}% of the corpus...")

        # Iterate through the randomized list of elements...
        for element in File.get_randomized_files_list(self.corpus_parent_dir):
            # ...and process the element only if we haven't process enough and it's a file.
            if to_process > 0 and File.is_file(element):
                self.__process_file(element)
                to_process -= 1

        Log.info(f"# Loaded {to_load} of {self.corpus_size} files.")

    def __init__(self, directory, load_percentage=DEFAULT_LOAD_PERCENTAGE,
                 language=DEFAULT_LANGUAGE, stop_words=DEFAULT_STOP_WORDS):
        # Set vars
        self.dataset = []
        self.corpus_parent_dir = directory
        self.language = language
        self.stop_words = stop_words
        self.load_percentage = load_percentage
        self.corpus_size = File.get_files_count(directory)

        # Load corpus from directory
        self.__load_corpus()
