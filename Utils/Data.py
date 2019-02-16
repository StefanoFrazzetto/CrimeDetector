from typing import List, Dict


class Email(object):

    @staticmethod
    def from_content(text):
        return Email(text).body

    def __init__(self, text):
        """
        Create an email object from a string.

        Source
        ------
        https://stackoverflow.com/questions/17874360/python-how-to-parse-the-body-from-a-raw-email-given-that-raw-email-does-not#32840516
        """
        import email
        msg = email.message_from_string(text)

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get_values('Content-Disposition'))

                if content_type == 'text/plain':
                    if 'attachment' not in content_disposition:
                        self.body = part.get_payload(decode=True)
                    else:
                        self.attachment = part.get_payload()

        else:
            self.body = msg.get_payload()

        self.raw = msg


class Text:

    @staticmethod
    def clean_email_body(email: Email):
        body = email.body
        return Text.clean(body)

    @staticmethod
    def clean(text):
        """
        Applies some pre-processing on the given text.
        Source: https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73

        Steps :
        - Removing HTML tags
        - Removing punctuation
        - Lowering text
        """
        import re

        # remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # remove the characters [\], ['] and ["]
        text = re.sub(r"\\", "", text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r"\"", "", text)

        # text to lowercase
        text = text.strip().lower()

        # replace punctuation characters with spaces
        filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, " ") for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)

        return text


class Time:
    @staticmethod
    def get_timestamp_millis():
        import time
        return int(round(time.time() * 1000))

    @staticmethod
    def millis_to_seconds(time1, time2):
        if time1 < time2:
            time1, time2 = time2, time1
        return (time1 - time2) / 1000


class DataStructures(object):
    @staticmethod
    def dictionary_list_to_dataframe(data: List[Dict]):
        """
        Convert a list of dictionaries to a Pandas DataFrame.
        :param data:
        :return:
        """
        import pandas as pd
        columns = data[0].keys()
        return pd.DataFrame(data, columns=columns)

    @staticmethod
    def list_chunks(l: List, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @staticmethod
    def merge_lists(list1: list, list2: list) -> list:
        return list1 + list2

    @staticmethod
    def merge_dicts(*dict_args):
        """
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        """
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result


class Hashing(object):
    @staticmethod
    def sha256_digest(data: str) -> str:
        """
        Return the digest value of the data, encoded in UTF-8, as a string of hexadecimal digits.
        :param data: the data to calculate the SHA256 digest for.
        :return: the digest value of the data
        """
        import hashlib
        data_hash = str(data).encode('utf-8')
        return hashlib.sha256(data_hash).hexdigest()


class Numbers(object):
    @staticmethod
    def get_formatted_percentage(partial, total):
        if partial == 0 or total == 0:
            ratio = 0
        else:
            ratio = partial / total
        percentage = ratio * 100
        return "{0:.2f}".format(percentage)

    @staticmethod
    def format_float(number: float, decimals: int = 2):
        stf = "{0:." + str(decimals) + "f}"
        return stf.format(number)
