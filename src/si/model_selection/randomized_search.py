#importação dos packages necessários:
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
    
    for parameter in hyperparameter_grid: #verifica se os hiperparâmetros existem no modelo fornecido
        if not hasattr(model, parameter): #se não existem
            raise AttributeError(f"The {model} does not have parameter {parameter}.") #mostra um AttributeError

    results = {'hyperparameters': [], 'scores': []} #inicia um dicionário results que armazenará os resultados da procura

    for _ in range(n_iter): #loop para criar diferentes combinações aleatórias de hiperparâmetros
        #gera uma combinação aleatória de hiperparâmetros baseada nos valores fornecidos no hyperparameter_grid:
        hyperparameters = {param: np.random.choice(values) for param, values in hyperparameter_grid.items()}

        for parameter, value in hyperparameters.items(): #define os hiperparâmetros gerados no modelo.
            setattr(model, parameter, value)

        #realiza a validação cruzada usando os hiperparâmetros atuais e obtém os scores:
        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        results['hyperparameters'].append(hyperparameters) #armazena os hiperparâmetros usados
        results['scores'].append(np.mean(scores)) #armazena a média dos scores obtidas com os hiperparâmetros

    best_index = np.argmax(results['scores']) #encontra o índice da combinação de hiperparâmetros com maior score
    results['best_hyperparameters'] = results['hyperparameters'][best_index] #armazena a melhor combinação de hiperparâmetros encontrada
    results['best_score'] = results['scores'][best_index] #armazena o score correspondente à melhor combinação de hiperparâmetros

    return results #devolve o dicionário results contendo as melhores combinações de hiperparâmetros e os scores correspondentes

