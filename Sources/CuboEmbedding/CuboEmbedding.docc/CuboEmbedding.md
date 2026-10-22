# ``CuboEmbedding``

CPU principal component analysis and k-means, with scikit-learn behaviour as the primary API.

Use ``PCA/fit(_:rows:columns:components:variant:)`` and ``KMeans/fit(_:rows:columns:initialCentroids:maxIterations:tolerance:isCancelled:)``
for scikit-learn 1.4.2 conventions. ``PCA/Variant/diffractionEmbeddingApp``, ``KMeans/fitSeeded(_:rows:columns:groups:seed:maxIterations:isCancelled:)``
and ``DiffractionFeatures`` reproduce mac4DSTEM's diffraction-embedding numerics as named options.
Settings and presentation are the caller's.

For data too large to hold, ``PCACovarianceAccumulator`` streams the mean and covariance block by block,
``PCA/fit(covariance:count:mean:components:)`` fits from it, and ``PCASignResolver`` gives the components scikit-learn's
signs from a second pass. ``DiffractionFeatures/binnedVector(pattern:rows:columns:binnedSize:implementation:)`` has an
Accelerate implementation, and ``KMeans`` takes an `isCancelled` closure.

## Topics

### Essentials

- <doc:Clustering-patterns>
- <doc:Scope>

### Principal components and k-means

- ``PCA``
- ``PCAResult``
- ``KMeans``
- ``KMeansResult``
- ``EmbeddingError``

### Streamed PCA

- ``PCACovarianceAccumulator``
- ``PCASignResolver``
- ``PCA/fit(covariance:count:mean:components:)``
- ``PCA/fit(_:components:)``

### Features

- ``DiffractionFeatures``
- ``DiffractionFeatures/Implementation``
