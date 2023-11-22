import numpy as np
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import Node, DecisionTreeClassifier
from collections import Counter
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
    def init(self, n_estimators = int, max_features=None, min_sample_split=2, max_depth=None, mode = 'gini', seed=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    
    def set_random_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)

    
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
        self.dataset = dataset
        n_samples, n_features = dataset.X.shape
        self.set_random_seed()

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
            X_bootstrap = dataset.X[sample_indices][:, feature_indices]
            y_bootstrap = dataset.y[sample_indices]

            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split,
                                          max_depth=self.max_depth,
                                          mode=self.mode)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append((feature_indices, tree))

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
            X_subset = dataset.X[:, tree.feature_indices]
            predictions = tree.predict(Dataset(X=X_subset))
            all_predictions.append(predictions)

        all_predictions = np.array(all_predictions).T
        return np.array([np.argmax(np.bincount(sample)) for sample in predictions])
    

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
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)
    
