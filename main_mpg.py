import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn import metrics

from utils import str2bool
from utils import print_info
from utils import plot_corr_matrix


def read_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path_train', type=str, default='datasets/auto-MPG/auto-MPG_tr.csv',
                        help='Path to the file containing the training set.')
    parser.add_argument('--dataset_path_test', type=str, default='datasets/auto-MPG/auto-MPG_tr.csv',
                        help='Path to the file containing the test set.')
    parser.add_argument('--val_percent', type=float, default=0.2,
                        help='Percentage of elements that will be used for the validation set.')
    parser.add_argument("--verbose", type=str2bool, default=True)

    args = parser.parse_args()

    return args


def fill_na(df: pd.DataFrame, column_label: str = 'HP',
            verbose: bool = False): #column_label seleziona la colonna HP

    if verbose:
        print("\nChecking for NaN/null values.")
        print(df.isnull().any()) #mi restituisce True se ho valori nulli, altrimenti false
    print("stampo database ")
    print(df)
    # Calcola la media della colonna column_label
    mean_val = df[column_label].mean()
    print(mean_val)
    # Sostituisci i valori nulli nella colonna column_label con la media
    df[column_label].fillna(mean_val, inplace=True) #Fill NA/NaN values using the specified method.
    print("------------")
    print(df)
    if verbose:
        print("\nNaN/null values after filling.")
        print(df.isnull().any())

    return df

'''
uesta è la definizione di una classe Python CarMPGPredictor.

Il costruttore __init__ della classe inizializza due attributi vuoti: self.preprocessing e self.regressor.

Il metodo fit è responsabile di addestrare il modello di regressione sui dati di addestramento. In questo metodo, si calcolano i parametri di trasformazione dei dati, si applica la trasformazione e si addestra il regressore sui dati trasformati.

Il metodo predict è responsabile di effettuare le previsioni sulle nuove osservazioni, applicando la stessa trasformazione usata nel metodo fit e usando il regressore addestrato.

Il metodo get_regressor_coeffs restituisce i coefficienti del regressore addestrato.

Il metodo print_metrics calcola tre metriche di valutazione del modello (Mean Absolute Error, Mean Squared Error e Root Mean Squared Error) e stampa i loro valori a schermo. Inoltre, se do_plot è impostato su True, genera un grafico che visualizza le predizioni e gli errori del modello.

In generale, la classe CarMPGPredictor sembra essere un modello di regressione basato sulla Ridge Regression per la predizione del consumo di carburante di un'automobile. Tuttavia, il codice presenta dei TODO, ovvero delle parti in cui il codice effettivo deve essere implementato per far funzionare correttamente la classe.'''


class CarMPGPredictor:
    ''''
Questo è il codice di una classe CarMPGPredictor che sembra essere un modello di regressione lineare utilizzando la regolarizzazione Ridge. In particolare, sembra che la classe sia destinata a prevedere il consumo di carburante di una macchina in base ad alcune sue caratteristiche.

Per utilizzare questa classe, dovresti:

    Creare un'istanza dell'oggetto CarMPGPredictor come model = CarMPGPredictor().
    Dividere i dati di training in X_train e y_train, dove X_train è una matrice di dati di input e y_train è un array di output (i.e., il consumo di carburante).
    Utilizzare il metodo fit(X_train, y_train) per addestrare il modello sui dati di training.
    Usare il modello addestrato per prevedere il consumo di carburante per nuovi dati utilizzando il metodo predict(X_test), dove X_test è una matrice di dati di input.
'''
    def __init__(self):
        # TODO: Scalamento delle feature usando la standardizzazione
        self.preprocessing = StandardScaler() #uso SKLEARN in particolare StandardScaler

        # TODO: istanzio la Ridge regression
        # La Ridge regression è un algoritmo di machine learning utilizzato per la regressione lineare con regolarizzazione L2. La regolarizzazione L2 è una tecnica comune utilizzata per prevenire il sovra-adattamento (overfitting) dei modelli di machine learning. L'obiettivo della regolarizzazione L2
        # è di penalizzare i coefficienti del modello che hanno valori elevati,
        # in modo che il modello preferisca coefficienti più piccoli.
        # istanzio la Ridge regression con un parametro di regolarizzazione alpha
        self.regressor = Ridge(alpha=1.0)
        #La regolarizzazione è una tecnica utilizzata in machine learning per ridurre l'overfitting (sovrapposizione) di un modello ai dati di addestramento.
        # L'overfitting si verifica quando il modello è troppo complesso e si adatta troppo bene ai dati di addestramento, ma non generalizza bene su nuovi dati.
        #La regolarizzazione introduce un termine di penalizzazione nell'obiettivo di minimizzazione della funzione di perdita del modello,
        # che influenza la scelta dei parametri del modello durante la fase di addestramento. Questo termine di penalizzazione limita i valori assunti dai parametri del modello,
        #'costringendoli ad assumere valori più piccoli. Ciò porta a modelli meno complessi e a una maggiore capacità di generalizzazione.
        #Ci sono diverse tecniche di regolarizzazione comunemente utilizzate, tra cui la regolarizzazione L1 e L2. La regolarizzazione L1 introduce una penalizzazione proporzionale alla somma dei valori assoluti dei parametri del modello,
        # mentre la regolarizzazione L2 introduce una penalizzazione proporzionale alla somma dei quadrati dei valori dei parametri del modello.

    def fit(self, X_train, y_train):
        # calcolo i parametri di trasformazione
        self.preprocessing.fit(X_train)

        # applico la transformazione
        X_train_transformed = self.preprocessing.transform(X_train)

        # fitto il regressore
        self.regressor.fit(X_train_transformed, y_train)
    ''' Il metodo predict() prende come input una matrice X di dati di input non visti dal modello e restituisce un array di valori predetti y_pred. Per fare una predizione, 
    il metodo applica la trasformazione ai dati di input utilizzando il metodo transform() della classe StandardScaler e quindi fa la predizione utilizzando il metodo 
    predict() della classe Ridge. Infine, il valore predetto viene restituito dal metodo. '''

    def predict(self, X):
        #faccio la trasfromazioni sulle X non del training test
        # applico la trasformazione
        X_transformed = self.preprocessing.transform(X)

        # effettuo la predizione
        y_pred = self.regressor.predict(X_transformed)
        return y_pred

    def get_regressor_coeffs(self):
        return self.regressor.coef_

    def print_metrics(self, y_val: np.array, y_pred: np.array,
                      do_plot: bool = True):

        print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))

        if do_plot:
            # TODO: realizzare un plot di visualizzazione che mostri le predizioni e gli errori del modello

            # realizzare un plot di visualizzazione che mostri le predizioni e gli errori del modello
            import matplotlib.pyplot as plt


            '''Il primo grafico mostra una dispersione dei valori predetti rispetto ai valori veri. Ogni punto nel grafico rappresenta una coppia di valori (y_val, y_pred) per un singolo esempio di test. La linea rossa diagonale indica la linea di uguaglianza tra i valori veri e i valori predetti.
            Il secondo grafico mostra la distribuzione degli errori del modello rispetto ai valori veri. Ogni punto nel grafico rappresenta una coppia di valori (y_val, errore), dove l'errore è la differenza tra il valore predetto e il valore vero. La linea rossa orizzontale indica la linea dello zero, cioè dove l'errore è nullo.
            In sintesi, questi grafici consentono di valutare visivamente quanto bene il modello sia in grado di predire i valori veri e come gli errori del modello siano distribuiti rispetto ai valori veri.'''

            errors = y_pred - y_val
            plt.figure(figsize=(10, 5))
            plt.scatter(y_val, y_pred)
            plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r')
            plt.xlabel('Valori veri')
            plt.ylabel('Valori predetti')
            plt.title('Predizioni vs. valori veri')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.scatter(y_val, errors)
            plt.plot([min(y_val), max(y_val)], [0, 0], 'r')
            plt.xlabel('Valori veri')
            plt.ylabel('Errori')
            plt.title('Errori vs. valori veri')
            plt.show()


if __name__ == '__main__':

    args = read_args()

    # load dataset
    df = pd.read_csv(args.dataset_path_train)
    # more from: http://archive.ics.uci.edu/ml/datasets/Auto+MPG
    # MPG è la variabile target

    if args.verbose:
        print("\nGeneric information about the dataset.")
        # Un'occhiata al dataset
        print_info(df)

    # TODO: modificare la funzione fill_na per gestire i valori nulli
    # Gestistico i valori nulli
    fill_na(df, verbose=args.verbose)

    # TODO: Sulla base della correlation matrix, scegliere le 4 features "migliori"
    # Plottare la correlation matrix, calcolata con pandas.
    # per scegliere le 4 features guardo la matrice di correlazione (l'immagine). devo scegliere le features che correlano
    #di più. mi concentro alla riga MPG e colonna MPG. e vedo le features del dataset quanto correlano con MPG. Devo
    # avere le features che sono più correlato con MPG (sono le 4 nere col valore più alto)
    plot_corr_matrix(df, verbose=args.verbose)
    features = ['CYL','DIS', 'HP', 'WGT']

    X, y = df[features].values, df['MPG'].values

    # TODO: usare la funzione train_test_split per dividere il dataset in train e val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CarMPGPredictor()

    ########### Training ######################
    model.fit(X_train, y_train)

    if args.verbose:
        print("\nCoefficients learned during ridge regression.")
        coeff_df = pd.DataFrame(model.get_regressor_coeffs(),
                                features, columns = ['Coefficient'])
        print(coeff_df)

    ########### Testing su train e val set. ######################
    y_pred = model.predict(X_train)
    print("\nPerformance on the training set.")
    model.print_metrics(y_train, y_pred, do_plot=True)
    '''
    # TODO: Misuro i risultati sul validation set usando MAE, MSE ed RMSE
    mae = mean_absolute_error(y_val,y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    print("MAE", mae)
    print("MSE", mse)
    print("RMSE", rmse)'''
    y_val_pred = model.predict(X_val)

    # Calcolare le metriche di valutazione del modello sul set di dati di validazione
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)

    # Stampa delle metriche di valutazione del modello sul set di dati di validazione
    print("Performance on the validation set:")
    print("MAE: {:.2f}".format(mae))
    print("MSE: {:.2f}".format(mse))
    print("RMSE: {:.2f}".format(rmse))
    ########### Testing sul testing set ######################

    # TODO: effettuare il test
    # 1) caricare il dataset (path in args.dataset_path_test)
    # 2) gestire i valori nulli
    # 3) selezionare le colonne giuste su cui fare il test
    # 4) effettuare la predizione
    # 5) stampare le metriche

