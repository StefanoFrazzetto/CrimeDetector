import pandas as pd
from typing import List, Dict


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
