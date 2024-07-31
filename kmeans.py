import os

import dotenv
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

dotenv.load_dotenv()

db_url = os.getenv("POSTGRES_URL") + "kmeans"

engine = create_engine(db_url)

# Fetch data from the users table
query = '''
    SELECT customer_id, name, email, phone_number FROM users;
'''

df = pd.read_sql_query(query, db_url)

df['name_hash'] = df['name'].apply(lambda x: hash(x) % 10**8)
df['email_hash'] = df['email'].apply(lambda x: hash(x) % 10**8)
df['phone_hash'] = df['phone_number'].apply(lambda x: hash(x) % 10**8)

# Extract the features for clustering
features = df[['name_hash', 'email_hash', 'phone_hash']]

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Reduce dimensions to 2D for visualization using PCA
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=df['cluster'], cmap='viridis', s=50, alpha=0.7)
plt.title('KMeans Clustering of Users')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
