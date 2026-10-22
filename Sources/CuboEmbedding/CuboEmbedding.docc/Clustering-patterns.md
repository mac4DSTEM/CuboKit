# Clustering patterns

An embedding turns many patterns into a few numbers each and groups similar ones. The kit does the two numerical steps on
row-major `[Double]` matrices, one row for each pattern. Reading patterns from a cube and turning each into a feature
vector is the caller's.

``PCA/fit(_:rows:columns:components:variant:)`` returns a ``PCAResult``: the `mean`, the `components` (one row each, in
descending order of variance), `explainedVariance`, `explainedVarianceRatio` and the `scores`, which are the data projected
on the components. ``PCA/transform(_:rows:result:)`` projects further rows on the same result.

``KMeans/fit(_:rows:columns:initialCentroids:maxIterations:tolerance:isCancelled:)`` runs Lloyd's algorithm from centroids
you give it, and returns a ``KMeansResult`` with a `labels` entry for every row, the `centroids`, the `inertia`, and whether
it `converged`.

```swift
import CuboEmbedding

// Six rows with three features: two groups, one near (0, 0, 0) and one near (5, 5, 5).
let data: [Double] = [0, 0.1, 0, 0.1, 0, 0.1, 0, 0, 0.2,
                      5, 5.1, 5, 5.1, 5, 5.1, 5, 5, 5.2]
let pca = try PCA.fit(data, rows: 6, columns: 3, components: 2)
let result = try KMeans.fit(pca.scores, rows: 6, columns: 2,
                            initialCentroids: Array(pca.scores[0..<2]) + Array(pca.scores[6..<8]))
print(result.labels)                                                    // [0, 0, 0, 1, 1, 1]
```

The default variant, ``PCA/Variant/scikitLearn``, follows scikit-learn 1.4.2, including its component signs.
``PCA/Variant/diffractionEmbeddingApp`` follows mac4DSTEM's own Float32 numerics. ``KMeans/fitSeeded(_:rows:columns:groups:seed:maxIterations:isCancelled:)``
chooses the first centroids from a seed, as the app does. scikit-learn's random seeding is not provided.

## When the patterns do not fit in memory

``PCACovarianceAccumulator`` adds the patterns block by block, so only one block is held. ``PCA/fit(_:components:)`` fits
from the accumulated covariance. The components then have the sign of the eigensolver, not scikit-learn's: pass a second
pass over the data through ``PCASignResolver`` and call `resolved()` for the scikit-learn signs.
