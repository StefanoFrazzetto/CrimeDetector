import re

from Classification.Data import Data, Email


class Text:

    @staticmethod
    def email_body_from_data(data: Data):
        text, _, _ = data.unpack()
        email = Email(text)
        return Text.clean_email_body(email)

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
