import re


class TextUtils:
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
