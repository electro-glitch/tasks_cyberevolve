## K-Means Clustering

**K-Means** is an **unsupervised clustering algorithm** that partitions a dataset into $K$ non-overlapping groups, or clusters, such that points within each cluster are as close as possible to the cluster’s center (centroid). It is widely used in applications like image compression, document categorization, customer segmentation, and general pattern discovery.


### Core Idea

The central goal of K-Means is to minimize the **within-cluster variance**, i.e., how far points in a cluster are from their centroid. The algorithm assumes that clusters are **spherical**, **equally sized**, and **linearly separable** in feature space.



### Mathematical Objective

Given a dataset $X = \{x_1, x_2, \dots, x_n\}$, where each point $x_i \in \mathbb{R}^d$, and a desired number of clusters $K$, the objective is to minimize the following cost function:

$$
J = \sum_{i=1}^{K} \sum_{x_j \in C_i} \|x_j - \mu_i\|^2
$$

Where:

* $C_i$ is the set of points assigned to cluster $i$,
* $\mu_i$ is the centroid (mean) of points in $C_i$,
* $\|x_j - \mu_i\|^2$ is the squared Euclidean distance.

The cost function is often referred to as the **Within-Cluster Sum of Squares (WCSS)**.


### The K-Means Algorithm (Lloyd’s Algorithm)

K-Means operates iteratively using the following steps:

1. **Initialization**: Select $K$ initial centroids $\{\mu_1, \mu_2, \dots, \mu_K\}$, typically chosen randomly or using K-Means++ for better stability.

2. **Assignment Step**:
   Each point $x_j$ is assigned to the cluster with the nearest centroid:

   $$
   \text{Assign } x_j \text{ to cluster } C_i \text{ where } i = \arg\min_{k} \|x_j - \mu_k\|^2
   $$

3. **Update Step**:
   Update each centroid to be the mean of the points in its cluster:

   $$
   \mu_i = \frac{1}{|C_i|} \sum_{x_j \in C_i} x_j
   $$

4. **Convergence Check**:
   Repeat steps 2 and 3 until the centroids stop changing significantly or until a maximum number of iterations is reached.



### Geometry of Clusters

K-Means partitions the space into **Voronoi cells**, where each region contains the points closest to one centroid. This implies:

* Decision boundaries are **linear (hyperplanes)**.
* Works well when clusters are **convex** and of **similar size**.



### Distance Metric

K-Means uses **squared Euclidean distance** by default:

$$
\text{dist}^2(x, \mu) = \sum_{i=1}^d (x_i - \mu_i)^2
$$

This ensures that the centroid update step (taking the mean) minimizes the cost function directly.



### Time Complexity

* **Per iteration**: $O(n \cdot K \cdot d)$
* **Total**: $O(n \cdot K \cdot d \cdot t)$, where $t$ is the number of iterations until convergence



## Key Hyperparameters

### 1. Number of Clusters ($K$)

* Must be chosen before running the algorithm.
* Common methods for estimating $K$:

  * **Elbow Method**: Plot WCSS vs $K$ and look for a point of inflection.
  * **Silhouette Score**: Measures how similar a point is to its cluster vs. others.
  * **Gap Statistic**: Compares clustering performance to that of a random uniform distribution.

### 2. Initialization Method (`init`)

* **Random**: Selects $K$ data points at random.
* **K-Means++**: Improves initialization by spreading out centroids:

  * First centroid is random.
  * Each subsequent centroid is chosen with probability proportional to its squared distance from the nearest existing centroid.

### 3. Number of Initializations (`n_init`)

* The algorithm is run `n_init` times with different initial centroids.
* The solution with the lowest WCSS is retained.
* Common default: `n_init = 10`.

### 4. Maximum Iterations (`max_iter`)

* Sets an upper limit on the number of iterations.
* Prevents infinite loops in non-converging runs.
* Typical value: `300`.



## Variants of K-Means

### K-Means++

A better initialization strategy that helps avoid poor local minima and accelerates convergence. Recommended over random initialization.

### Mini-Batch K-Means

Uses small, randomly sampled batches of data to update centroids, rather than the full dataset. Offers:

* Faster runtime on large datasets
* Slightly noisier but still effective results

### Bisecting K-Means

A **hierarchical version** of K-Means:

* Starts with one cluster containing all data
* Recursively splits the largest cluster into two using standard K-Means
* Continues until $K$ clusters are formed

### Fuzzy K-Means (Fuzzy C-Means)

Allows points to belong to multiple clusters with degrees of membership. The objective function becomes:

$$
J_m = \sum_{i=1}^{K} \sum_{j=1}^{n} u_{ij}^m \|x_j - \mu_i\|^2
$$

* $u_{ij} \in [0,1]$: degree of membership of point $x_j$ in cluster $i$
* $m > 1$: fuzziness coefficient



## Evaluation Metrics

| Metric           | Description                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------- |
| WCSS (Inertia)   | Total intra-cluster variance. Lower is better.                                              |
| Silhouette Score | Balance of cohesion and separation. Ranges from -1 to 1.                                    |
| Davies–Bouldin   | Measures average similarity between each cluster and its most similar one. Lower is better. |



## Advantages

* Simple and fast for low to moderate dimensions
* Efficient on large datasets
* Guaranteed convergence (though to a local minimum)
* Easy to interpret



## Disadvantages

* Requires pre-specification of $K$
* Sensitive to:

  * Initial centroid positions
  * Outliers
  * Scale of features
* Performs poorly on:

  * Non-spherical clusters
  * Clusters of varying densities or sizes



## Applications

* Market and customer segmentation
* Vector quantization (e.g., for image compression)
* Grouping sensor readings or log files
* Document or topic clustering

---


## Hierarchical Clustering

**Hierarchical Clustering** is an unsupervised learning algorithm that builds a hierarchy of nested clusters without needing to predefine the number of clusters. It works by either **merging smaller clusters** into larger ones (agglomerative), or **splitting a large cluster** into smaller ones (divisive), producing a tree-like structure known as a **dendrogram**.

The agglomerative method is the more common and computationally practical approach.

---

### How Agglomerative Clustering Works

1. **Start** with each data point as its own cluster.
2. **Compute distances** between all clusters.
3. **Merge the two closest clusters** based on a chosen linkage criterion.
4. **Update distances** between the new cluster and the rest.
5. **Repeat** until only one cluster remains.

The merging sequence forms a binary tree that records how clusters are combined at increasing distance thresholds.



### Dendrogram and Cluster Formation

A **dendrogram** is a hierarchical tree where:

* Leaves represent individual data points.
* Internal nodes represent merges of clusters.
* The vertical axis denotes the distance (or dissimilarity) at which merges occur.

To extract final clusters:

* Cut the dendrogram at a chosen height (distance threshold).
* The resulting connected components below the cut define the clusters.

This allows forming flat clusters post hoc, without committing to a specific number in advance.



### Linkage Methods (Cluster-to-Cluster Distance)

When merging clusters, the following **linkage criteria** define how inter-cluster distances are measured:

* **Single Linkage**:

  $$
  d(A, B) = \min_{a \in A, b \in B} \|a - b\|
  $$

  Uses the closest pair between two clusters. Can lead to long, chain-like clusters.

* **Complete Linkage**:

  $$
  d(A, B) = \max_{a \in A, b \in B} \|a - b\|
  $$

  Uses the farthest pair. Encourages compact, spherical clusters.

* **Average Linkage**:

  $$
  d(A, B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} \|a - b\|
  $$

  Averages all pairwise distances. More stable and balanced.

* **Ward’s Method**:

  $$
  \Delta(A, B) = \frac{|A||B|}{|A| + |B|} \|\mu_A - \mu_B\|^2
  $$

  Merges the pair of clusters that results in the **minimum increase in total intra-cluster variance**. Works only with Euclidean distance. Tends to produce well-separated clusters of similar size.


### Distance Metrics

Pairwise distance is calculated using one of the following, depending on data type and context:

* **Euclidean**: default for most cases and required for Ward’s method
* **Manhattan**: for grid-based layouts or robust behavior
* **Cosine**: for high-dimensional or sparse data (e.g. text)
* **Hamming**: for binary or categorical variables



### Hyperparameters and Decisions

Even though hierarchical clustering is non-parametric in theory, several **decisions must be made** during setup and post-processing:

#### 1. **Distance Metric**

* Determines how similarity between individual points is computed.
* Common choices: Euclidean, Manhattan, Cosine, Hamming.

#### 2. **Linkage Method**

* Affects the shape, size, and granularity of resulting clusters.
* Typical options: single, complete, average, Ward’s (default with `scikit-learn` if Euclidean is used).

#### 3. **Cut-Off Strategy**

Controls the formation of final, flat clusters from the dendrogram:

* **Distance Threshold** (`t`): Clusters are formed by cutting the tree at a fixed dissimilarity.
* **Number of Clusters** (`k`): Cut the tree to yield a specified number of clusters.

The threshold can be manually set or inferred using visual inspection or metrics like inconsistency coefficients.

#### 4. **Data Scaling**

As clustering is distance-based, **feature scaling (standardization or normalization)** is essential. This acts as a preprocessing hyperparameter.



### Divisive Clustering (Less Common)

The **divisive** variant works top-down:

* Start with all data in one cluster.
* Recursively split clusters based on internal dissimilarity (e.g., using K-Means or PCA projections).
* Continue until only singletons remain or a stopping criterion is met.

Although conceptually appealing, divisive clustering is rarely used due to its higher computational cost.



### Complexity

* **Agglomerative**: $O(n^2 \log n)$ with optimized implementations (e.g., nearest-neighbor chains)
* **Divisive**: Potentially $O(2^n)$, not scalable

Hierarchical clustering is not suited for very large datasets without approximation methods.


### Evaluation Metrics

| Metric                        | Description                                                                                         |
| ----------------------------- | --------------------------------------------------------------------------------------------------- |
| **Cophenetic Correlation**    | Measures how well the dendrogram preserves pairwise distances. Higher is better.                    |
| **Silhouette Score**          | Measures how well a point fits into its cluster vs. others. Ranges from -1 to 1.                    |
| **Dunn Index**                | Ratio of minimum inter-cluster distance to maximum intra-cluster distance. Higher is better.        |
| **Inconsistency Coefficient** | Compares height of a cluster link to heights of previous links. Helps choose where to cut the tree. |
| **Adjusted Rand Index**       | Compares predicted clustering to a ground truth (if available). 1 = perfect match.                  |





### Advantages

* Does not require predefining the number of clusters
* Produces interpretable dendrograms
* Can reveal multi-level cluster structures
* Flexible via choice of distance and linkage methods



### Limitations

* High computational cost for large $n$
* Sensitive to noise and outliers
* Merges are greedy and irreversible — early mistakes cannot be undone
* Different linkage strategies may yield very different results



### Applications

* Biological taxonomy (e.g., phylogenetic trees)
* Gene expression analysis
* Document/topic clustering
* Network and community detection
* Market segmentation (when data has nested structure)

---


## Principal Component Analysis (PCA)

**PCA** is a **linear dimensionality reduction** technique that transforms high-dimensional data into a lower-dimensional space by projecting it onto a new set of axes called **principal components**. These new axes capture the directions of maximum variance in the data.



### Goal

To reduce the number of dimensions (features) in the dataset while retaining as much **variability (information)** as possible.


### Key Idea

PCA finds a new coordinate system such that:

* The **first axis** captures the largest variance in the data.
* The **second axis** captures the next largest variance, orthogonal (perpendicular) to the first.
* And so on...

These new axes are the **principal components**, and the data is **projected** onto them.



### Notation

Let:

* $X \in \mathbb{R}^{n \times d}$: Input data matrix

  * $n$: Number of data points (rows)
  * $d$: Number of features (columns)
* $x_i \in \mathbb{R}^d$: A single data point (a row vector)
* $\mu \in \mathbb{R}^d$: The mean vector of all features
* $\Sigma \in \mathbb{R}^{d \times d}$: Covariance matrix
* $v_1, v_2, \dots, v_k \in \mathbb{R}^d$: Top $k$ eigenvectors (principal components)
* $\lambda_1, \lambda_2, \dots, \lambda_k$: Corresponding eigenvalues (variance explained)



### Steps of PCA (with Symbol Explanation)

#### 1. Center the Data

$$
X_c = X - \mu
$$

* Subtract the mean of each feature so that data is centered at the origin.
* $\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$ is the mean vector.

#### 2. Compute Covariance Matrix

$$
\Sigma = \frac{1}{n} X_c^\top X_c
$$

* $X_c^\top$: Transpose of the centered data.
* Covariance measures how two features vary together.
* $\Sigma_{ij}$ is large if features $i$ and $j$ vary together.

#### 3. Eigen Decomposition

$$
\Sigma v_i = \lambda_i v_i
$$

* Solve for eigenvectors $v_i$ and eigenvalues $\lambda_i$.
* $v_i$: Direction of the $i$-th principal component.
* $\lambda_i$: Variance explained along that direction.
* Eigenvectors are sorted so that $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d$.

#### 4. Project Data

Select the top $k$ eigenvectors and **project** the original data:

$$
Z = X_c \cdot V_k
$$

* $V_k \in \mathbb{R}^{d \times k}$: Matrix of top $k$ eigenvectors (each column is a principal component).
* $Z \in \mathbb{R}^{n \times k}$: Transformed data in reduced $k$-dimensional space.



### What Does "Projection" Mean Here?

In geometry, to project a vector onto another vector (or axis) means to **drop a perpendicular** from the point onto the line (axis) and use the foot of that perpendicular as the projection.

Mathematically:

$$
\text{Projection of } x \text{ onto } v = (x \cdot v) v
$$

So in PCA:

* Each data point $x$ is **projected onto the new axes** (principal components).
* The dot product $x \cdot v_i$ gives the coordinate of $x$ along $v_i$.
* The set of these new coordinates forms a **lower-dimensional representation** of $x$.



### Summary of Properties

* PCA always finds **orthogonal** (uncorrelated) axes.
* The total variance is preserved and redistributed across the new axes.
* It's sensitive to feature scale → standardization is important before applying PCA.
* PCA is **rotation invariant** but not **scale invariant**.



## t-SNE (t-Distributed Stochastic Neighbor Embedding)

**t-SNE** is a **nonlinear**, **probabilistic** dimensionality reduction algorithm primarily used for **visualizing** high-dimensional data in 2 or 3 dimensions. Unlike PCA, it focuses on **preserving local structure**, i.e., nearby points in high-D should stay nearby in low-D.



### High-Level Idea

* Model **pairwise similarity** in high-D space using probabilities.
* Do the same in low-D space.
* **Minimize the difference** between these two distributions.



### Notation and Explanation

Let:

* $x_i, x_j \in \mathbb{R}^d$: High-dimensional data points
* $y_i, y_j \in \mathbb{R}^2$: Low-dimensional embeddings
* $p_{ij}$: Joint probability that $x_i$ and $x_j$ are neighbors in high-D
* $q_{ij}$: Joint probability that $y_i$ and $y_j$ are neighbors in low-D
* $\sigma_i$: Gaussian bandwidth for point $x_i$



### Step 1: Compute High-D Similarities

For each pair of points:

$$
p_{j|i} = \frac{\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2} \right)}{\sum_{k \neq i} \exp\left(-\frac{\|x_i - x_k\|^2}{2\sigma_i^2} \right)}
$$

This is a **softmax** over distances: closer points get higher probabilities.

Symmetrize:

$$
p_{ij} = \frac{p_{i|j} + p_{j|i}}{2n}
$$

So now $p_{ij} \in [0, 1]$, and $\sum_{i \ne j} p_{ij} = 1$



### Step 2: Compute Low-D Similarities

Use a **Student t-distribution with 1 degree of freedom** (a Cauchy distribution):

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \ne l} (1 + \|y_k - y_l\|^2)^{-1}}
$$

The t-distribution’s heavy tails help **separate distant points**, avoiding crowding in 2D.



### Step 3: Minimize KL Divergence

The cost function is the **Kullback–Leibler divergence**:

$$
\text{KL}(P \| Q) = \sum_{i \ne j} p_{ij} \log \left( \frac{p_{ij}}{q_{ij}} \right)
$$

This is minimized via **gradient descent**, pulling similar points together and pushing dissimilar ones apart.



### Perplexity: Controlling Locality

Perplexity is a **tunable hyperparameter** that determines the size of the neighborhood considered when computing $p_{j|i}$.

Defined as:

$$
\text{Perplexity}(P_i) = 2^{H(P_i)}, \quad H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}
$$

Interpretation:

* Perplexity \~ effective number of nearest neighbors
* The algorithm searches for a value of $\sigma_i$ (bandwidth) for each point that achieves the desired perplexity
* Affects how local the similarity computation is

#### Guidelines

| Perplexity   | Behavior                                    |
| ------------ | ------------------------------------------- |
| 5–15         | Very local; more sensitive to fine clusters |
| 30 (default) | Balanced local/global tradeoff              |
| 50+          | Broad, more global; may blur tight clusters |

Try multiple values to ensure robust patterns.



### Other Important Parameters

| Hyperparameter       | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| `perplexity`         | Controls neighborhood size; affects similarity estimation     |
| `learning_rate`      | Gradient step size. Too small = slow; too large = noise       |
| `n_iter`             | Number of optimization steps; usually 1000+                   |
| `early_exaggeration` | Temporarily amplifies attraction to form tight clusters early |
| `init`               | How low-D points are initialized (`random`, `pca`)            |
| `method`             | Barnes-Hut or FFT approximations for large datasets           |



### Important Notes

* t-SNE is **non-convex** → different runs can yield different layouts
* Sensitive to scale → standardize data
* Only for visualization → not useful for downstream models directly
* Not ideal for preserving global structure



## Linear vs Nonlinear Structure

### What Does "Linear" Mean in Dimensionality Reduction?

* A **linear relationship** implies that a feature or component can be expressed as a **linear combination** of others.
* **PCA** assumes that the data lies on or near a **flat subspace (hyperplane)** in the high-dimensional space.

### What is a Nonlinear Structure?

* Data that lies on a **curved manifold** embedded in higher dimensions (e.g., spiral, concentric circles) is **nonlinear**.
* PCA cannot "unroll" or unfold such structures because its projections are linear.

#### Example:

* A 3D spiral projected onto a 2D plane by PCA still looks like a spiral.
* **t-SNE**, however, can unfold this spiral and place similar points close together in 2D.



## Why t-SNE Outperforms PCA on Nonlinear Data

t-SNE does not project the data linearly. Instead, it:

* Converts distances into probabilities
* Preserves **local neighborhoods**
* Allows arbitrary, nonlinear warping of space to emphasize local clusters

As a result, t-SNE often reveals **natural groupings** and **cluster structure** in data that PCA would miss entirely.

However, this comes with trade-offs:

* t-SNE does **not preserve global structure** (e.g., distances between clusters)
* Results can vary due to randomness and are not interpretable in terms of axes or features



## PCA vs. t-SNE: Comparison Table

| Feature/Property     | **PCA**                                        | **t-SNE**                                         |
| -------------------- | ---------------------------------------------- | ------------------------------------------------- |
| Type                 | Linear transformation                          | Nonlinear, probabilistic mapping                  |
| Preserves            | Global structure (variance, directions)        | Local structure (neighborhoods)                   |
| Projection           | Onto orthogonal axes (principal components)    | Into flexible space; no fixed basis               |
| Assumption           | Linearity, Euclidean geometry                  | No linear assumption                              |
| Output Stability     | Deterministic (given the same input)           | Stochastic (varies unless random seed is fixed)   |
| Interpretability     | High (axes have feature loadings)              | Low (axes are abstract; no physical meaning)      |
| Use Case             | Preprocessing, compression, data understanding | Visualization of clusters in complex datasets     |
| Handles Nonlinearity | Not effectively                                            |    Effectively                                              |
| Computational Cost   | Low $O(nd^2)$                                  | High $O(n^2)$; Barnes-Hut reduces to $O(n\log n)$ |
| Works with New Data? |  (projection formula available)               |  (must retrain with entire dataset)              |
| Main Hyperparameter  | `n_components`                                 | `perplexity`, `learning_rate`, `n_iter`, etc.     |


* **Use PCA** when:

  * Data relationships are roughly linear
  * Interpretability matters
  * Data preprocessing or compression is the goal

* **Use t-SNE** when:

  * Visualizing high-dimensional data
  * Clusters are expected to be nonlinear or tangled
  * Preserving **local relationships** is more important than global distances

---


# Autoencoders

An **Autoencoder** is a neural network trained to replicate its input at the output, typically via an **information bottleneck**. It learns to encode the input into a lower-dimensional representation and then decode it back, forcing the model to extract and preserve the most essential features of the data.

Autoencoders are **unsupervised**: they do not require labels during training. Their primary use is **representation learning** — learning useful, compressed features from raw input data.


## Core Working

Autoencoders work by minimizing the **reconstruction error** between an input $x \in \mathbb{R}^d$ and its output $\hat{x} \in \mathbb{R}^d$, which is obtained after passing through two parts:

### 1. **Encoder**: Feature Compression

The encoder maps the input to a **latent vector** $z \in \mathbb{R}^k$, where $k < d$:

$$
z = f_\theta(x)
$$

This mapping is typically implemented via one or more fully connected (or convolutional) layers:

$$
z = \sigma(W_e x + b_e)
$$

* $W_e \in \mathbb{R}^{k \times d}$: weight matrix
* $b_e \in \mathbb{R}^k$: bias vector
* $\sigma$: activation function (e.g., ReLU)

The encoder **learns a nonlinear transformation** that compresses the input into a lower-dimensional space — the **latent space**.



### 2. **Latent Space**

The latent space is a lower-dimensional vector space that represents the **internal encoding** of the input. Ideally:

* Points that are close in input space are also close in latent space
* The latent space captures the **intrinsic structure** or **degrees of variation** of the data

In practice, this means that similar data points (e.g., two handwritten digits) are mapped to nearby points in latent space, and that **interpolating** between two codes produces valid intermediate representations.

---

### 3. **Decoder**: Reconstruction

The decoder maps the latent representation back to input space:

$$
\hat{x} = g_\phi(z)
$$

This is again a neural network:

$$
\hat{x} = \sigma(W_d z + b_d)
$$

* $W_d \in \mathbb{R}^{d \times k}$, $b_d \in \mathbb{R}^d$
* The decoder **inverts** the encoding, approximating the original input from compressed data



### 4. **Training Objective: Reconstruction Loss**

Autoencoders are trained to minimize reconstruction error over the dataset. The most common loss functions are:

* **Mean Squared Error (MSE)** for continuous input:

  $$
  \mathcal{L}_{\text{MSE}}(x, \hat{x}) = \frac{1}{d} \sum_{i=1}^d (x_i - \hat{x}_i)^2
  $$

* **Binary Cross-Entropy (BCE)** for binary or normalized input:

  $$
  \mathcal{L}_{\text{BCE}}(x, \hat{x}) = - \sum_{i=1}^d \left[ x_i \log \hat{x}_i + (1 - x_i)\log (1 - \hat{x}_i) \right]
  $$

This loss is minimized via **stochastic gradient descent** (SGD) or its variants (e.g., Adam), using backpropagation to adjust the encoder and decoder parameters simultaneously.



## Role of the Bottleneck

The bottleneck — the latent layer — is what differentiates an autoencoder from a plain identity function. By limiting the capacity of the latent representation:

* The model is forced to **compress** information
* Redundancy and noise are discarded
* Only features that are important for reconstruction are retained

Without a bottleneck, the network would trivially learn to copy the input (i.e., a near identity function).



## Hyperparameters

Here are the essential hyperparameters that affect how autoencoders learn and what they learn.

### 1. **Latent Dimension (`k`)**

* Defines how much compression occurs
* Affects how abstract the encoding is
* Too small → underfitting (not enough capacity)
* Too large → overfitting (copies input with no abstraction)

### 2. **Activation Functions**

* Typically ReLU or tanh in the encoder
* Decoder uses:

  * **Sigmoid** if input is normalized to \[0, 1] (BCE loss)
  * **Linear** if input is unbounded (MSE loss)

The choice of activation affects convergence and the range of output values.

### 3. **Loss Function**

* Chosen based on data type:

  * MSE → continuous input (e.g., grayscale image pixels)
  * BCE → binary or probability-like input (e.g., 0–1 normalized images)

### 4. **Learning Rate**

* Controls the step size in gradient descent
* Must be tuned carefully — too high causes divergence, too low slows learning
* Typical range: $10^{-3}$ to $10^{-5}$

### 5. **Model Depth and Width**

* More layers increase capacity but risk overfitting
* Deep models can learn hierarchical features
* Symmetric encoder-decoder structures are common, but not required



## Real-World Applications


* **Image Denoising**: Removing noise from images by learning the mapping from noisy input to clean reconstruction.
* **Anomaly Detection**: Training on normal data, anomalies are identified by high reconstruction error.
* **Data Compression**: Latent representation can be stored/transmitted instead of full input.
* **Feature Extraction**: Encoder output can be used as learned features for classification or clustering.
* **Visualization**: Latent space (if 2D or 3D) can be directly visualized; higher-dimensional latent space can be visualized via t-SNE or PCA.

---


## DBSCAN: Hyperparameters

DBSCAN's performance is governed primarily by two hyperparameters:

* $\varepsilon$ (epsilon): the **neighborhood radius**
* $\text{MinPts}$: the **minimum number of neighbors** needed to define a **core point**

These parameters control how **local density** is measured, and thus how clusters are defined.



### 1. **Epsilon ( $\varepsilon$ )**

#### Definition:

$\varepsilon$ defines the **maximum distance** between two points for one to be considered part of the other’s **neighborhood**.

Mathematically, for a point $x \in \mathbb{R}^d$, its ε-neighborhood is:

$$
\mathcal{N}_\varepsilon(x) = \{ y \in \mathcal{D} \mid \|x - y\| \leq \varepsilon \}
$$

#### Role:

This parameter determines what constitutes a **"local region"**. Only points within this region are counted for density estimation.

#### Theoretical Effect:

* **Smaller $\varepsilon$** implies **higher resolution**: clusters must be denser to be detected.
* **Larger $\varepsilon$** implies **lower resolution**: sparser groups of points may be merged.

#### Impact on Clustering:

| ε Range   | Outcome                                                             |
| --------- | ------------------------------------------------------------------- |
| Too small | Few core points; many points classified as noise                    |
| Moderate  | Clusters discovered as dense regions; noise filtered                |
| Too large | Merged clusters; the entire dataset may be grouped into one cluster |

#### Choosing ε:

**k-distance plot** is a widely used heuristic:

* For each point, compute the distance to its k-th nearest neighbor.
* Sort these distances in ascending order and plot them.
* Look for the **"elbow" point** — a sharp change in slope.
* The y-coordinate of this elbow approximates a good value for $\varepsilon$.

Notes:

* $k = \text{MinPts} - 1$ is commonly used for this plot.
* The elbow represents a boundary between dense and sparse regions.

#### Geometry Sensitivity:

* In high-dimensional spaces, distances tend to **concentrate** (i.e., distances become similar across pairs), making ε less informative.
* Always **standardize** or **normalize** features before applying DBSCAN.



### 2. **Minimum Points (MinPts)**

#### Definition:

The **minimum number of points (including the center point itself)** required in an ε-neighborhood for a point to be considered a **core point**.

Formally:

$$
x \text{ is a core point} \iff |\mathcal{N}_\varepsilon(x)| \geq \text{MinPts}
$$

#### Intuition:

MinPts defines how dense a region must be to qualify as a cluster.

#### Guideline for Selection:

| Dimensionality $d$ | Suggested MinPts |
| ------------------ | ---------------- |
| Low (e.g., 2–3)    | 4 to 6           |
| Moderate (5–10)    | 6 to 10          |
| High (>10)         | ≥10              |

General rule:

$$
\text{MinPts} \geq d + 1
$$

This ensures that clusters are **statistically significant**, especially as dimensions increase.

#### Effect on Behavior:

| MinPts Value | Behavior                                                                          |
| ------------ | --------------------------------------------------------------------------------- |
| Too low      | Algorithm becomes sensitive to noise and spurious patterns                        |
| Moderate     | Proper density estimation; clusters form where local neighborhoods are consistent |
| Too high     | Many points classified as noise; smaller clusters may be missed                   |

#### Practical Considerations:

* **Larger MinPts** increases cluster stability but risks missing valid but sparse clusters.
* **Smaller MinPts** increases sensitivity to noise and outliers.



### 3. **Interaction Between ε and MinPts**

The two parameters must be **tuned jointly**:

* **Small ε + Large MinPts**: Very conservative; requires tightly packed data; few or no clusters.
* **Large ε + Small MinPts**: Too permissive; merges distinct clusters and reduces noise filtering.

In general:

* Set MinPts based on data dimensionality.
* Use a k-distance plot to determine ε based on that MinPts.



### 4. **Effect of Distance Metric**

The distance metric (used to define $\|x - y\|$) directly affects the ε-neighborhood. DBSCAN supports:

* **Euclidean** (default): For numerical features.
* **Manhattan**: Alternative for grid-structured or absolute difference-based data.
* **Cosine**: For high-dimensional, sparse data (e.g., text, TF-IDF vectors).
* **Minkowski**, **Hamming**, etc.

Choosing an appropriate metric is essential, especially for **non-Euclidean domains**.



### 5. **Other Implementation-Level Parameters**

| Parameter   | Description                                                                       |
| ----------- | --------------------------------------------------------------------------------- |
| `metric`    | Distance function (e.g., `'euclidean'`, `'cosine'`)                               |
| `algorithm` | Nearest neighbor search strategy: `'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'` |
| `leaf_size` | Leaf size for tree structures; affects query efficiency                           |
| `n_jobs`    | Number of CPU cores to use for parallel computation (if applicable)               |



