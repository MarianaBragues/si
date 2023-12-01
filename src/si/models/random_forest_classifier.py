import numpy as np
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import Node, DecisionTreeClassifier
from si.metrics.accuracy import accuracy

class RandomForestClassifier:
    """
    Random Forest is an ensemble machine learning technique that combines multiple decision trees to improve prediction 
    accuracy and reduce overfitting.

    
    parameters:
    -----------
    n_estimators: 
        number of decision trees to use
    
    max_features:
        maximum number of features to use per tree
    
    min_sample_split:
        minimum samples allowed in a split

    max_depth:
        maximum depth of the trees

    mode:
        impurity calculation mode (gini or entropy)

    seed:
        random seed to use to assure reproducibility

        
    estimated parameters:
    ---------------------
    trees: list
        the trees of the random forest and respective features used for training (initialized as an empty list)
    """
    def __init__(self, n_estimators=100, max_features=None, min_sample_split=2, max_depth=None, mode = 'gini', seed=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    
    def set_random_seed(self):
        """
        Set a random seed for NumPy

        Returns
        -------
        A random value
        """
        if self.seed is not None:
            np.random.seed(self.seed)

    def get_bootstrap_dataset(self, dataset: Dataset):
        """
        Generate a bootstrap dataset with randomly sampled instances and features from the original dataset. 
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: 
            The fitted model
        """
        n_samples, n_features = dataset.shape()

        indices = np.random.choice(n_samples, n_samples, replace=True)
        features = np.random.choice(n_features, self.max_features, replace=False)
        
        return Dataset(X=dataset.X[indices][:, features], y=dataset.y[indices])


    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Train the decision trees of the random forest

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RandomForest
            The fitted model
        """
        self.set_random_seed()
        n_features = dataset.shape()[1]
        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features

        if self.max_depth is None:
            self.max_depth = 10

        for _ in range(self.n_estimators):
            bootstrap_dataset = self.get_bootstrap_dataset(dataset)
            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split,
                                          max_depth=self.max_depth,
                                          mode=self.mode)
            tree.fit(bootstrap_dataset)
            self.trees.append((bootstrap_dataset.features, tree))

        return self
    

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predicts the labels using the ensemble models

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        #self.dataset = dataset  
        all_predictions = []

        for _, tree in self.trees:
            X_subset = dataset.X[:, tree.feature_idx]
            predictions = tree.predict(Dataset(X=X_subset))
            all_predictions.append(predictions)

        all_predictions = np.array(all_predictions).T
        return np.array([np.argmax(np.bincount(sample)) for sample in all_predictions])
    

    def score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy between predicted and real labels

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the RandomForest on

        Returns
        -------
        random_forest: float
            The Mean Square Error of the model
        """
        return accuracy(dataset.y, self.predict(dataset))
    
