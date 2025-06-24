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
   (\text{reach-dist}_k(A, B) = \max(\text{distance}(A, B), \text{k-distance}(B))
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



---




## KD Tree (k-dimensional tree)

A **KD Tree** is a binary tree used to **organize points in a k-dimensional space**. It’s mainly used to **speed up nearest neighbor searches** — like in KNN, LOF, or clustering tasks.

Instead of checking distances to all points, a KD Tree breaks up the space and lets us skip big chunks when searching — just like binary search, but in multiple dimensions.

---

### Basic Idea

Think of a KD Tree as a way to **cut up the space** into smaller and smaller regions, so we can quickly find where a point lies or which point is closest to another.

For example, in 2D space, the KD Tree might first split by the x-axis, then y-axis, then x again, and so on.


### How It Works

1. **Choose a splitting dimension**
   Start with the first dimension (e.g. x-axis), and sort all points by that feature.

2. **Pick the median point**
   This becomes the root node. It splits the space into two halves.

3. **Recursively build left and right subtrees**

   * For the left side: use points less than the median.
   * For the right side: use points greater than the median.
   * Alternate the splitting dimension at each level (e.g., x → y → x → y...).

4. **Continue until each point is placed in a node**
   The result is a binary tree where each node splits space in one direction.


#### Example (2D Points)

Given points:

```
(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)
```

* First split: x-axis → median is (5,4)
* Left subtree: points with x < 5
* Right subtree: x > 5
* Next level: y-axis → split left and right again
* Continue recursively

This creates a tree that cuts the space into rectangles — helpful for quick searches.


### Nearest Neighbor Search

When searching for the closest point to a query:

* Go down the tree to the "most likely" leaf node.
* Backtrack and check only **branches that might contain a closer point**.
* This avoids having to check every point.

KD Tree makes nearest-neighbor queries much faster, especially in **low dimensions**.

---

### Advantages

* **Fast nearest-neighbor search** (better than brute force)
* Great for **2D–10D** data (like geolocation, images, basic ML datasets)
* Used in **KNN, LOF, clustering, etc.**



### Disadvantages

* Doesn’t scale well to **high dimensions** (e.g., 100D)
* Needs to be **balanced** for best performance
* Rebuilding is costly if data changes often



### Use Cases

* Finding nearby items on a map
* Speeding up KNN classification
* Faster similarity search in recommendation systems
* Clustering spatial data (like image pixels or geolocation)

---

## Ball Tree

Ball Tree is a binary tree data structure that speeds up nearest neighbor search by organizing data points into nested balls (hyperspheres) instead of axis-aligned splits (like KD Trees).

The core idea: if a whole group of points is far away from your target, you can skip all of them at once — which saves a ton of time.


### Analogy: Russian Dolls

Think of the dataset like a stack of Russian dolls:

* The largest doll (root node) wraps all points.
* Inside it, two smaller dolls split the space further.
* Each smaller doll contains tighter clusters.
* Eventually, we reach the smallest dolls, which just wrap a few individual data points.

Each "doll" here is a ball, which has:

* A center (the average point — a.k.a. the centroid)
* A radius (how far the farthest point is from the center)

---

### How It Works 

1. **Start with all the data points**

2. **Find the centroid**

   * The centroid is the **mean of each feature** (dimension).
   * In 2D, it's just the average of all x’s and all y’s.
   * In high dimensions, do this for every feature (column).

3. **Compute the radius**

   * This is the **maximum Euclidean distance** from the centroid to any point in the current subset.

4. **Split into two child balls**

   * Choose two points that are far apart (like seeding a 2-cluster split).
   * Assign each remaining point to whichever of the two is closer.
   * For each group:

     * Find its **centroid**
     * Compute its **radius**
     * These become **child balls**

5. **Repeat recursively**

   * Keep splitting until each ball contains very few points (a leaf).
   * Now you have a tree of **nested balls**, each wrapping a local group of points.



### Nearest Neighbor Search (How It Prunes)

When searching for the nearest neighbor to a query point:

1. Traverse the tree to find the most promising leaf.
2. Track the **shortest distance** so far (call it `D_best`).
3. On the way back up:

   * For any unexplored ball, check:

     $$
     \text{distance(query, center) - radius} > D_\text{best}
     $$

   * If true → skip the entire ball (no point inside can be closer).

   * If false → go deeper into that ball and check its points.

This lets Ball Trees **ignore entire clusters of points** that are guaranteed to be too far away.


### Why Radius and Centroid Vary

* Each ball covers only its own subset of points
* The **centroid** and **radius** are recalculated at each level
* Tightly packed groups = small radius
* Looser groups = large radius

So the Ball Tree essentially adapts to the data’s local structure.

---

### Advantages

* Works better than KD Trees in **high-dimensional data**
* Prunes large portions of the dataset quickly
* Great for **KNN**, **LOF**, **clustering**, and **similarity search**

### Disadvantages

* Tree construction is more complex
* Slightly slower than KD Tree in very low dimensions
* Doesn’t handle dynamic updates well (requires rebuilding)


### Real-World Use Cases

* Fast K-Nearest Neighbors for image or text embeddings
* Outlier detection (used in LOF)
* Recommender systems
* Clustering high-dimensional data

---


## Common Distance Metrics in ML


### 1. **Euclidean Distance**

**Formula:**

For points `A = (x₁, x₂, ..., xₙ)` and `B = (y₁, y₂, ..., yₙ)`:

$$
\text{Euclidean}(A, B) = \sqrt{ \sum_{i=1}^{n} (x_i - y_i)^2 }
$$

**Intuition:**
Straight-line (as-the-crow-flies) distance between two points.

**Use when:**

* Dimensions have the **same scale**
* We care about **geometric distance**



### 2. **Manhattan Distance** (L1 Norm, "Taxicab" distance)

**Formula:**

$$
\text{Manhattan}(A, B) = \sum_{i=1}^{n} |x_i - y_i|
$$

**Intuition:**
How distance is covered on a city grid — up/down, left/right only.

**Use when:**

* We want a **more robust**, less sensitive-to-outliers distance
* In high-dimensional or sparse data



### 3. **Minkowski Distance** (Generalized distance)

**Formula (for p ≥ 1):**

$$
\text{Minkowski}(A, B) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
$$

Special cases:

* `p = 1`: Manhattan Distance
* `p = 2`: Euclidean Distance
* `p → ∞`: Chebyshev Distance

**Use when:**

* We want a tunable distance metric (`p` controls sensitivity)



### 4. **Chebyshev Distance** (L∞ Norm)

**Formula:**

$$
\text{Chebyshev}(A, B) = \max_{\text{i}} |x_i - y_i|
$$

**Intuition:**
Only considers the **largest** difference across dimensions.

**Use when:**

* We care about the **worst-case** deviation
* Often used in chess, logistics



### 5. **Cosine Distance** (used with cosine similarity)

**Formula (distance = 1 - similarity):**

$$
\text{Cosine}(A, B) = 1 - \frac{A \cdot B}{||A|| \cdot ||B||}
$$

Where `A ⋅ B` is the dot product, and `||A||` is the magnitude (Euclidean norm).

**Intuition:**
Focuses on **direction**, not magnitude — how similar two vectors "point".

**Use when:**

* In **text**, NLP, embeddings (like word2vec, BERT)
* We want similarity regardless of vector size



### 6. **Hamming Distance** (for binary/categorical data)

**Formula:**

$$
\text{Hamming}(A, B) = \sum_{i=1}^{n} [x_i \neq y_i]
$$

(Counts how many positions differ between the two vectors)

**Use when:**

* Data is **categorical**, binary, or strings
* Used in error correction, fingerprinting
