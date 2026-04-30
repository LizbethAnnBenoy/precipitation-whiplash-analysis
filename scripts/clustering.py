from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_clustering(df):
    whiplash_events = df[df['is_whiplash'] == True].copy()

    if len(whiplash_events) > 10:
        cluster_features = ['precipitation_mm', 'ivt', 'drought_severity_score', 'dry_land_memory']
        X_cluster = whiplash_events[cluster_features].values

        scaler = StandardScaler()
        X_cluster_scaled = scaler.fit_transform(X_cluster)

        optimal_k = 3

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        whiplash_events['cluster'] = kmeans.fit_predict(X_cluster_scaled)

        cluster_means = whiplash_events.groupby('cluster')['precipitation_mm'].mean().sort_values()
        cluster_labels = {cluster_means.index[0]: 'Mild',
                         cluster_means.index[1]: 'Moderate',
                         cluster_means.index[2]: 'Severe'}

        whiplash_events['severity_class'] = whiplash_events['cluster'].map(cluster_labels)

        whiplash_events.to_csv('whiplash_events_clustered.csv', index=False)

    return whiplash_events
