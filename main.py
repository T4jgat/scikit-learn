import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing, cluster

df = pd.read_csv('force2020_data_unsupervised_learning.csv', index_col='DEPTH_MD')

df.dropna(inplace=True)

scaler = preprocessing.StandardScaler()

df[['RHOB_T', 'NPHI_T', 'GR_T', 'PEF_T', 'DTC_T']] = (
    scaler.fit_transform(df[['RHOB', 'NPHI', 'GR', 'PEF', 'DTC']]))


def optimize_kmeans(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    fig = plt.subplots(figsize=(10, 5))

    plt.plot(means, inertias, 'o-')
    plt.xlabel('Num of clusters')
    plt.ylabel('Inertia')

    plt.grid(True)
    plt.show()


# optimize_kmeans(df[['RHOB_T', 'NPHI_T']], 10)

kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df[['NPHI_T', 'RHOB_T']])

df['kmeans_3'] = kmeans.labels_

plt.scatter(x=df['NPHI'], y=df['RHOB'], c=df['kmeans_3'])
plt.xlim(-0.1, 1)
plt.ylim(3, 1.5)

print(df)
plt.show()