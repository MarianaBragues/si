import numpy as np

from si.data.dataset import Dataset


class SelectPercentile:
    def __init__(self, score_func, percentile=25):
        """
        Initializes the select percentile.
        :param score_func:
        :param percentile:
        """
        self.score_func = score_func
        self.percentile = percentile
        self.f_value = None
        self.p_value = None

    def fit(self, dataset):
        """
        Calculates the score for each feature.
        :param dataset: Dataset object.
        :return: SelectPercentile object.
        """
        self.f_value, self.p_value = self.score_func(dataset)

        return self

    def transform(self, dataset):
        """
        Selects the percentile best features with the highest score until the percentile is reached.
        :param dataset: Dataset object.
        :return: Dataset object.
        """
        if self.f_value is None:
            raise Exception("You must fit the select percentile before transform the dataset.")

        features_percentile = int(len(dataset.features_names) * self.percentile / 100)  # calculates the number of
        # features to be selected based on the percentile value (e.g. 50% of 10 features = 5 features)
        indexes = np.argsort(self.f_value)[-features_percentile:]
        best_features = dataset.x[:, indexes]
        best_features_names = [dataset.features_names[i] for i in indexes]

        return Dataset(best_features, dataset.y, best_features_names, dataset.label_name)

    def fit_transform(self, dataset):
        """
        Calculates the score for each feature and selects the percentile best features.
        :param dataset: Dataset object.
        :return: Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)