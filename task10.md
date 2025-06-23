## Isolation Forest (iForest)

**Isolation Forest** is an **unsupervised machine learning algorithm** used mainly for **anomaly detection**. It works on the principle that **anomalies are rare and different**, so they can be **isolated faster** than normal points.

---

### Basic Idea

Unlike other models that try to learn patterns in the data (like how normal data behaves), iForest does the opposite.
It randomly splits data and sees how quickly a point can be isolated. The assumption is:

If a data point gets separated (or "isolated") **in fewer steps**, it's probably an **anomaly**.


### How It Works

1. **Random Sampling**
   iForest picks random small subsets of the original dataset. Usually, the size is around 256 points, but this can vary.

2. **Building Isolation Trees**
   For each sample, it builds a tree by:

   * Randomly selecting a feature (column)
   * Choosing a random split value between the min and max of that feature
   * Splitting the data recursively until each point gets isolated or a max depth is reached

3. **Path Length Calculation**
   For each data point, it calculates how many splits it took to isolate it in a tree. This number is called the path length.

4. **Scoring Anomalies**
   If a point has a short average path length across all trees, it's considered more likely to be an anomaly.
   If it takes more steps to isolate, it's probably normal.


### Anomaly Score

The anomaly score is calculated using this formula:

$$
s(x, n) = 2^{-\frac{h(x)}{c(n)}}
$$

Where:

* $h(x)$: average path length for point $x$
* $c(n)$: average path length for a dataset of size $n$
* Score near 1 → likely outlier
* Score near 0.5 → likely normal

---

### Advantages

* Fast and scalable, even on large datasets
* No need to scale data or assume any distribution
* Handles high-dimensional data well

### Disadvantages

* Not ideal for very small datasets
* Struggles when anomalies aren't clearly separated
* Less interpretable than simple distance-based methods



### Real-World Use Cases

* Fraud detection
* Network intrusion detection
* Detecting faulty sensors or health records


---

## Local Outlier Factor (LOF)

**Local Outlier Factor (LOF)** is an **unsupervised anomaly detection algorithm** that finds outliers by analyzing the **local density** of data points.

The main idea is: if a point lies in a **much sparser region** compared to its nearby neighbors, it’s likely to be an **outlier**.

---

### Built on KNN: But Smarter

LOF builds upon the idea of **K-Nearest Neighbors (KNN)**.

In KNN, we check how close a point is to its `k` nearest neighbors to make predictions or measure similarity.

But LOF takes this idea further:

Instead of just checking how *close* a node is to its neighbors, it also asks:

**"Is the area around the node as crowded as the areas around its neighbors?"**

This shift from measuring just distance to also comparing **local densities** is what makes LOF effective for detecting **local anomalies** — outliers that are only unusual in their specific region of the dataset.


### What is `k`?

* `k` is the number of nearest neighbors to consider.
* LOF uses this to define the "local neighborhood" around each point.
* A typical value is `k = 20`, but this depends on your data.
* Smaller `k` makes LOF more sensitive to small variations; larger `k` smooths things out.

---

### How LOF Works

1. Find the `k` nearest neighbors of each point (just like in KNN)

2. Calculate **Reachability Distance**
   For each neighbor, define:

   $$
   \text{reach-dist}_k(A, B) = \max(\text{distance}(A, B), \text{k-distance}(B))
   $$

   This avoids very small distances dominating the calculation.

3. **Calculate Local Reachability Density (LRD)**
   For a point $A$, compute the **inverse of the average reachability distance** to its neighbors:

   $$
   \text{LRD}(A) = \left( \frac{1}{\sum \text{reach-dist}(A, \text{neighbor}) / k} \right)
   $$

   This gives an idea of how **crowded or sparse** the area around $A$ is.

4. **Compute the LOF Score**
   Now compare $A$’s density to its neighbors' densities:

   $$
   \text{LOF}(A) = \frac{\sum \text{LRD(neighbors of A)}}{\text{LRD}(A) \times k}
   $$

   
### Interpreting LOF Scores

| LOF Score | Interpretation                               |
| --------- | -------------------------------------------- |
| ≈ 1       | Similar to neighbors → normal point          |
| > 1       | Less dense than neighbors → possible outlier |
| ≫ 1       | Much less dense → strong outlier             |

---

### Advantages

* Can detect **outliers in complex, clustered datasets**
* Doesn’t rely on global assumptions — works locally
* Especially good when **data has regions of varying density**

### Disadvantages

* Slower on large datasets due to distance calculations
* Choosing the right `k` is important for accurate results
* Slightly harder to interpret than simpler methods

### Real-World Applications

* Finding unusual customer behavior
* Identifying faulty sensors in IoT
* Detecting intrusions or fraud in logs and networks
