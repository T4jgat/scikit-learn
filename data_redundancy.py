from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import dotenv
import os

dotenv.load_dotenv()

db_url = os.getenv("DATABASE_URL") + "customer_order_data_redundancy_test"

engine = create_engine(db_url)
query_customers = '''
    SELECT * FROM customers
'''
query_orders = '''
    SELECT * FROM orders
'''

customers_df = pd.read_sql(query_customers, engine)
orders_df = pd.read_sql(query_orders, engine)

features = customers_df[['customerid', 'customername']].copy()
features = pd.concat([
    features,
    orders_df[['customerid', 'customername']].rename(columns={'customername': 'ordercustomername'})
], axis=1)

features['customername'] = features['customername'].astype('category').cat.codes
features['ordercustomername'] = features['ordercustomername'].astype('category').cat.codes

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(scaled_features)

features['cluster'] = clusters

for cluster_num in set(clusters):
    print(f"\nCluster {cluster_num}:")
    cluster_data = features[features['cluster'] == cluster_num]
    print(cluster_data[['customerid', 'customername', 'ordercustomername']].drop_duplicates())

pca = PCA(n_components=2)
features_pca = pca.fit_transform(scaled_features)

plt.figure(figsize=(10, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=features['cluster'], cmap='viridis', s=50, alpha=0.7)
plt.title('KMeans Clustering of Users')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

