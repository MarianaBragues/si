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


#Tests:
if __name__ == '__main__':
    #criar um conjunto de dados de exemplo
    from si.models.logistic_regression import LogisticRegression
    from si.metrics.rmse import rmse
    np.random.seed(42)
    X = np.random.rand(100, 3)  #matriz de 100 amostras e 3 features
    theta_true = np.array([3, 1.5, -2])  #coeficientes verdadeiros
    noise = 0.1 * np.random.randn(100)  
    y = X.dot(theta_true) + noise  #gerar os valores y usando os coeficientes e adicionando noise

    #criar um Dataset
    dataset = Dataset(X, y)

    #definir o modelo (LogisticRegression) para a procura aleatória de hiperparâmetros
    model = LogisticRegression()

    #define a grade de hiperparâmetros para a procura
    hyperparameter_grid = {
        'learning_rate': (0.01, 0.1, 0.5),
        'epochs': (100, 200, 300),
        'batch_size': (32, 64, 128)
    }

    #realiza a procura aleatória de hiperparâmetros
    search_results = randomized_search_cv(model, dataset, hyperparameter_grid, scoring=rmse, cv=5, n_iter=10)

    #exibe os resultados da procura
    print("Best Hyperparameters:", search_results['best_hyperparameters'])
    print("Best Score (MSE):", search_results['best_score'])