import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile:
    def __init__(self, score_func = f_classification, percentile : int = 50):
        #define o construtor com os parâmetros iniciais
        """
        Select features with the highest F value up to the specified percentile.

        Parameters
        ----------
        score_func: callable, default = f_classification
            Variance analysis function. Function taking dataset and returning a pair of arrays (scores, p_values)

        percentile: int, default = 50
            Percentile for selecting features
        """
        self.score_func = score_func #calcula algum tipo de pontuação ou valor para as features, by default é f_classification
        self.percentile = percentile #percentil usado para selecionar features, by default é 50
        self.F = None
        self.p = None
        #as variáveis F e p são inicializadas como None
    

    def fit(self, dataset: Dataset) -> 'SelectPercentile': 
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
        self.F, self.p = self.score_func(dataset) #recorre à função score_func, utilizando o conjunto de dados fornecido
        #e calcula os valores de F e p para cada feature no conjunto de dados
        return self #devolve o objeto SelectPercentile depois de calcular F e p
    
    def transform(self, dataset: Dataset) -> Dataset:
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
        total_feat = len(dataset.features) #calcula o número total de features do dataset
        wanted_feat = int(total_feat*self.percentile/100) #determina o número desejado de features com base no percentil 
        #especificado na inicialização da classe SelectPercentile
        sorted_feat = np.argsort(self.F)[-wanted_feat:] #ordena os valores de F calculados durante o fit e seleciona os 
        #índices das features correspondentes aos maiores valores de F até ao número desejado (wanted_feat)
        best_feat = dataset.X[:, sorted_feat] #seleciona as features originais do dataset com base nos índices das features 
        #selecionadas
        names_feat = [dataset.features[i] for i in sorted_feat] #obtém os nomes das features selecionadas usando os índices
        #das features ordenadas
        return Dataset(X=best_feat, y=dataset.y, features=names_feat, label=dataset.label) #devolve um novo dataset com as
        #features selecionadas (best_feat), preservando as labels (dataset.y), os nomes das features selecionadas (names_feat) 
        #e o nome da label (dataset.label).
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
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
        self.fit(dataset) #chama o método fit previamente definido na classe SelectPercentile, que calcula os valores de F e p
        #para as features do conjunto de dados
        return self.transform(dataset) #chama o método transform para selecionar as features com base nos valores de F 
        #calculados pelo método fit. Devolve um novo dataset contendo apenas as features mais relevantes, conforme 
        #determinado pelo método transform
    
#Testes:    
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