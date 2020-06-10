from typing import Union

import numpy as np
from sklearn.preprocessing import LabelEncoder


class RobustLabelEncoder(LabelEncoder):
    def __init__(self, unseen_value: Union[str, int, float] = 0):

        """
        This class extends LabelEncoder functionality to the case when you assume
        that new unseen categories can be used as inputs to the model
        :param unseen_value: value to replace unseen values and encode them
        """

        super(RobustLabelEncoder, self).__init__()
        self.unseen_value = unseen_value

    def fit(self, y):
        return super(RobustLabelEncoder, self).fit(list(y) + [self.unseen_value])

    def transform(self, y):
        if isinstance(y, np.ndarray):
            y = np.where(~np.isin(y, self.classes_), self.unseen_value, y)
        else:
            y = [i if i in self.classes_ else self.unseen_value for i in list(y)]

        return super(RobustLabelEncoder, self).transform(y)
