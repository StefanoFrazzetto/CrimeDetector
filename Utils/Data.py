import email
import re
import time
from typing import List, Dict

import pandas as pd


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
        msg = email.message_from_string(text)

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))

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
        return int(round(time.time() * 1000))

    @staticmethod
    def millis_to_seconds(time1, time2):
        if time1 < time2:
            time1, time2 = time2, time1
        return (time1 - time2) / 1000


class DataConverter(object):
    @staticmethod
    def dictionary_list_to_dataframe(data: List[Dict]):
        """
        Convert a list of dictionaries to a Pandas DataFrame.
        :param data:
        :return:
        """
        columns = data[0].keys()
        return pd.DataFrame(data, columns=columns)

    @staticmethod
    def list_chunks(l: List, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]