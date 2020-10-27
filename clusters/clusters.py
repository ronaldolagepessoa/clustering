import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, OPTICS, AffinityPropagation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import plotly.express as px



class KMeansCluster:
    
    def __init__(self, DataFrame, n_clusters=8, eliminate_columns=None):
        self.df = DataFrame.copy()
        self.n_clusters = n_clusters
        if eliminate_columns is not None:
            self.df.drop(eliminate_columns, axis=1, inplace=True)
        self.df.dropna(axis=0, inplace=True)
        self.columns = list(self.df.columns)
        
        self.onehot = OneHotEncoder(sparse=False, drop="first")
        X_bin = self.onehot.fit_transform(self.df.select_dtypes(include=['object']))

        self.mmscaler = MinMaxScaler()
        X_num = self.mmscaler.fit_transform(self.df.select_dtypes(exclude=['object']))

        X_all = np.append(X_num, X_bin, axis=1)
        self.model = KMeans(n_clusters=self.n_clusters)
        self.df['cluster'] = self.model.fit_predict(X_all)
    
    def plot_parallel_categories(self):
        df1 = self.df.copy()
        df1.cluster = df1.cluster.astype('object')
        columns = df1.select_dtypes(include=['object']).columns
        return px.parallel_categories(df1.sort_values('cluster'), dimensions=columns, 
        title='Proporção dos atributos categóricos')

    def plot_sunburst(self):
        df1 = self.df.copy()
        df1.cluster = df1.cluster.astype('object')
        columns = df1.select_dtypes(include=['object']).columns
        df1 = df1[columns]
        df1['freq'] = 1
        return px.sunburst(df1, path=np.flip(columns), values='freq', 
        title='Proporção dos atributos categóricos')

    def plot_bar(self):
        return px.bar(self.df.groupby('cluster').mean().stack().reset_index(name='média'), 
        x='cluster', y='média', color='level_1', title='Média dos atributos numéricos')

    def plot_box(self):
        df1 = self.df.copy()
        columns = df1.select_dtypes(exclude=['object']).columns
        df1 = df1[columns]
        df1 = df1.set_index('cluster').stack().reset_index(name='valor')
        return px.box(df1, x='cluster', y='valor', color='level_1',  notched=True, 
        title='Distribuição dos atributos numéricos')

    def predict(self, lista):
        df = pd.DataFrame({column: [item] for column, item in zip(self.df.columns, lista)})
        X_bin = self.onehot.transform(df.select_dtypes(include=['object']))
        X_num = self.mmscaler.transform(df.select_dtypes(exclude=['object']))
        X_all = np.append(X_num, X_bin, axis=1)
        return self.model.predict(X_all)[0]




if __name__ == '__main__':
    df = pd.read_csv('https://raw.githubusercontent.com/ronaldolagepessoa/data_science/master/dados/airbnb_ny2.csv')
    model = KMeansCluster(df, eliminate_columns=['nome', 'bairro', 'latitude', 'longitude'])
    #nome,distrito,bairro,latitude,longitude,tipo_quarto,preco,minimo_noites,numero_reviews,numero_reservas,disponibilidade_anual
    #Clean & quiet apt home by the park,Brooklyn,Kensington,40.64749000000001,-73.97237,Quarto privado,149,1,9,6,365
    # plot = model.plot_sunburst() 
    # plot.show()
    print(model.predict(['Brooklyn', 'Quarto privado', 149, 1, 9, 6, 365]))
    print(model.columns)

    