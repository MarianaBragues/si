from typing import Dict, Tuple, Callable, Any

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model,
                         dataset: Dataset,
                         hyperparameter_grid: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 3,
                         n_iter: int = 10) -> Dict[str, Tuple[str, Any]]:
    
    #  checks if parameters exist in the model
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"The {model} does not have parameter {parameter}.")

    results = {'hyperparameters': [], 'scores': []}

    #  sets n_iter hyperparameter combinations
    for _ in range(n_iter):
        hyperparameters = {param: np.random.choice(values) for param, values in hyperparameter_grid.items()}

        for parameter, value in hyperparameters.items():
            setattr(model, parameter, value)

        # performs cross_validation with the combination
        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        results['hyperparameters'].append(hyperparameters)
        results['scores'].append(np.mean(scores))

    # stores the hyperparameter combination and the obtained scores
    best_index = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_index]
    results['best_score'] = results['scores'][best_index]

    return results