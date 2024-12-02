import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

def plot_scatter(df):
    for species in df["Species"].unique():
        subset = df[df["Species"] == species]
        plt.scatter(subset["SepalWidthCm"], subset["PetalLengthCm"], label=species)
    plt.title("Sepal Length vs Sepal Width")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.show()

def plot_pairplot(df):
    sns.pairplot(df, hue="Species", diag_kind="hist", height=2.5)
    plt.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=16)
    plt.show()

def perform_pca_and_plot(df):
    features = df.columns
    x = df[features[1:4]]
    y = df["Species"]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    pca_df = pd.DataFrame(x_pca, columns=["PC1", "PC2"])
    pca_df["Species"] = y
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

    for species in pca_df["Species"].unique():
        subset = pca_df[pca_df["Species"] == species]
        plt.scatter(subset["PC1"], subset["PC2"], label=species)
    plt.title("PCA of Iris Dataset")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Species")
    plt.show()

    return pca_df

def plot_binary_decision_boundary(pca_df):
    binary_df = pca_df.copy()
    binary_df["BinarySpecies"] = (binary_df["Species"] == "Iris-setosa").astype(int)

    clf_binary = LogisticRegression()
    clf_binary.fit(binary_df[["PC1", "PC2"]], binary_df["BinarySpecies"])

    x_min, x_max = binary_df["PC1"].min() - 1, binary_df["PC1"].max() + 1
    y_min, y_max = binary_df["PC2"].min() - 1, binary_df["PC2"].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = clf_binary.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(binary_df["PC1"], binary_df["PC2"], c=binary_df["BinarySpecies"], cmap=plt.cm.Paired)
    plt.title("Binary Classification Decision Boundary", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

def plot_multi_class_decision_boundary(pca_df):
    X_pca_features = pca_df[["PC1", "PC2"]].values
    y_labels = pca_df["Species"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)

    clf = LogisticRegression()
    clf.fit(X_pca_features, y_encoded)

    x_min, x_max = X_pca_features[:, 0].min() - 1, X_pca_features[:, 0].max() + 1
    y_min, y_max = X_pca_features[:, 1].min() - 1, X_pca_features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

    for species, label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        subset = pca_df[pca_df["Species"] == species]
        plt.scatter(subset["PC1"], subset["PC2"], label=species)

    plt.title("Decision Boundary with PCA Features", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Species")
    plt.show()

# Example usage:
# Load the dataset (ensure 'df' is properly loaded)
# df = pd.read_csv('datasets/Iris.csv')

# plot_scatter(df)
# plot_pairplot(df)
# pca_df = perform_pca_and_plot(df)
# plot_binary_decision_boundary(pca_df)
# plot_multi_class_decision_boundary(pca_df)
