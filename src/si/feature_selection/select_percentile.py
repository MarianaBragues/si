import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile:
    def __init__(self, score_func = f_classification, percentile : int = 50):
        """
        Select features with the highest F value up to the specified percentile.

        Parameters
        ----------
        score_func: callable, default = f_classification
            Variance analysis function. Function taking dataset and returning a pair of arrays (scores, p_values)

        percentile: int, default = 50
            Percentile for selecting features
        """
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
    
    def fit(self, dataset: Dataset):
        """
        It fits SelectPercentile to compute the F scores and p-values.
        Estimates the F and p values for each feature using the scoring_func.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """

        self.F, self.p = self.score_func(dataset)
        return self
    
    def transform(self, dataset: Dataset):
        """
        It transforms the dataset by selecting the highest F value up to the specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            The transformed Dataset object
        """
        total_feat = len(dataset.features)   #total number of features
        wanted_feat = int(total_feat*self.percentile/100)   #number of features based on percentile
        sorted_feat = np.argsort(self.F)[-wanted_feat:]   #sorts the F values
        best_feat = dataset.X[:, sorted_feat]
        names_feat = [dataset.features[i] for i in sorted_feat]
        return Dataset(X=best_feat, y=dataset.y, features=names_feat, label=dataset.label)
    
    def fit_transform(self, dataset: Dataset):
        """
        It fits SelectPercentile and transforms the dataset by selecting the features with the 
        highest F value up to the specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the features with the highest F value.
        """
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector_percentile = SelectPercentile(percentile=50)
    selector_percentile.fit(dataset)
    dataset = selector_percentile.transform(dataset)
    print(dataset.features)