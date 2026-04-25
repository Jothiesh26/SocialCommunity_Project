def run_analysis():

    import mysql.connector
    import networkx as nx
    import community as community_louvain
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # ---------------- DATABASE CONNECTION ----------------
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="user",
        database="social_community"
    )

    cursor = conn.cursor()

    # ---------------- FETCH DATA USING JOIN ----------------
    query = """
    SELECT p1.person_name, p2.person_name, r.group_type
    FROM relationships r
    JOIN person p1 ON r.person1_id = p1.person_id
    JOIN person p2 ON r.person2_id = p2.person_id
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    # ---------------- BUILD GRAPH ----------------
    G = nx.Graph()

    for person1, person2, group_type in rows:
        G.add_edge(person1, person2, group=group_type)

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    # ---------------- COMMUNITY DETECTION ----------------
    partition = community_louvain.best_partition(G)

    # ---------------- AI MODEL SECTION ----------------

    features = []
    labels = []

    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    for node in G.nodes():
        degree = G.degree(node)
        clustering = nx.clustering(G, node)
        centrality = degree_centrality[node]
        between = betweenness[node]
        close = closeness[node]

        features.append([degree, clustering, centrality, between, close])
        labels.append(partition[node])

    X = np.array(features)
    y = np.array(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    cv_scores = cross_val_score(model, X, y, cv=5)
    cv_accuracy = cv_scores.mean()

    # ---------------- MODULARITY ----------------
    modularity = community_louvain.modularity(partition, G)
    num_communities = len(set(partition.values()))

    # ---------------- SAVE COMMUNITY CSV ----------------
    with open("community_output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Person", "Community"])
        for person, comm_id in partition.items():
            writer.writerow([person, comm_id])

    # ---------------- SAVE GRAPH IMAGE FOR UI ----------------
    pos = nx.spring_layout(G, seed=42)
    colors = [partition[node] for node in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set3)
    plt.title("Community Detection using Louvain")

    # Save inside static folder
    plt.savefig("static/graph.png")
    plt.close()

    cursor.close()
    conn.close()

    # ---------------- RETURN RESULTS TO FLASK ----------------
    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "modularity": round(modularity, 4),
        "accuracy": round(accuracy, 4),
        "cv_accuracy": round(cv_accuracy, 4),
        "num_communities": num_communities
    }