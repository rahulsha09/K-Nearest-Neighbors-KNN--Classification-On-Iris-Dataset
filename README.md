# K-Nearest Neighbors (KNN) Classification On Iris Dataset

This project applies the K-Means clustering algorithm to the classic Iris dataset. Choose a classification dataset and normalize features, Use KNeighborsClassifier from sklearn, Experiment with different values of K, Evaluate model using accuracy, confusion matrix, Visualize decision boundaries.

## 🎯 Project Overview

This project demonstrates a complete workflow for unsupervised clustering:
1.  Loading and preprocessing the data.
2.  Determining the optimal number of clusters (`k`) using the Elbow Method.
3.  Training a K-Means model with the optimal `k`.
4.  Visualizing the resulting clusters and their centroids.
5.  Evaluating the quality of the clusters using the Silhouette Score.

## 🛠️ Methodology

1.  **Data Scaling**: The features are standardized using `StandardScaler` to ensure that each feature contributes equally to the distance-based calculations of the K-Means algorithm.

2.  **Elbow Method**: To find the best value for `k`, the Within-Cluster Sum of Squares (WCSS) is calculated for `k` values from 1 to 10. A line plot of WCSS vs. `k` is generated. The "elbow" of the curve, where the rate of decrease slows, indicates the optimal `k`.

3.  **K-Means Training**: A K-Means model is trained on the scaled data using the optimal `k` found from the Elbow Method (which is 3 for this dataset).

4.  **Visualization**: A 2D scatter plot is created to visualize the clusters based on sepal length and sepal width. Data points are color-coded by their assigned cluster, and the final cluster centroids are marked with stars.

## 📈 Results

-   **Optimal Clusters**: The Elbow Method clearly indicated that the optimal number of clusters for the Iris dataset is **k=3**.
    

-   **Cluster Visualization**: The final visualization shows three distinct and well-separated groups of flowers, corresponding closely to the three actual species.
    

-   **Evaluation**: The clustering achieved a **Silhouette Score of approximately 0.45**, indicating that the clusters are reasonably dense and well-separated.

## Explanations

Import Libraries:

pandas: For data manipulation and loading the CSV file.

matplotlib.pyplot and seaborn: For creating visualizations like the elbow plot and the final cluster scatter plot.

sklearn.cluster.KMeans: The primary class for implementing the K-Means algorithm.

sklearn.preprocessing.StandardScaler: To standardize the features, ensuring each feature contributes equally to the distance calculations.

sklearn.metrics.silhouette_score: An intrinsic metric to evaluate the quality of the formed clusters.

Load and Prepare Data:

The Iris.csv dataset is loaded into a pandas DataFrame.

We create our feature matrix X by selecting only the columns with measurements (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm). The non-feature Id and the ground-truth label Species are excluded.

Using StandardScaler, we scale the features. This is a crucial preprocessing step for K-Means because it is a distance-based algorithm. Scaling ensures that features with larger ranges (like Sepal Length) do not dominate the clustering process over features with smaller ranges (like Sepal Width).

Find Optimal 'k' (Elbow Method):

The K-Means algorithm requires us to specify the number of clusters, k. The Elbow Method is a popular heuristic for finding the optimal k.

We loop through a range of k values (1 to 10) and, for each k, we train a K-Means model.

We store the WCSS (Within-Cluster Sum of Squares) for each model. WCSS is the sum of squared distances between each data point and its assigned cluster's center (centroid). scikit-learn provides this value via the .inertia_ attribute.

We then plot k against WCSS. The plot typically looks like an arm. The point where the rate of decrease sharply slows down forms an "elbow," which suggests the optimal k. For the Iris dataset, this is clearly at k=3.

Train K-Means Model:

With the optimal k identified as 3, we instantiate the KMeans class with n_clusters=3.

init='k-means++' is a smart initialization technique that helps the algorithm converge better than random initialization.

random_state=42 ensures reproducibility.

The .fit_predict() method trains the model on the scaled data and returns the cluster label for each data point.

Visualize the Clusters:

To visualize the results, we create a scatter plot. Since the data has four dimensions, we choose two features for the x and y axes (SepalLengthCm and SepalWidthCm).

Each point is colored based on the cluster label assigned by our model.

We also plot the final cluster centroids as yellow stars. Since the model was trained on scaled data, we must use scaler.inverse_transform() to convert the centroid coordinates back to the original scale before plotting them.

Evaluate the Model:

In real-world unsupervised tasks, we lack ground-truth labels. Therefore, we use intrinsic evaluation metrics.

The Silhouette Score is one such metric. It calculates for each sample how close it is to its own cluster compared to other clusters. A score near +1 indicates well-defined clusters, a score near 0 indicates overlapping clusters, and a score near -1 indicates that samples might have been assigned to the wrong clusters. Our score of ~0.45 is respectable and indicates reasonably good cluster separation.
