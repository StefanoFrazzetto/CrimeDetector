from typing import List, Dict

import pandas as pd


class DataTools(object):
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
