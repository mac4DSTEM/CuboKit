# ``CuboEmbedding``

CPU principal component analysis and k-means, with scikit-learn behaviour as the primary API.

Use ``PCA/fit(_:rows:columns:components:variant:)`` and ``KMeans/fit(_:rows:columns:initialCentroids:maxIterations:tolerance:)``
for scikit-learn 1.4.2 conventions. ``PCA/Variant/diffractionEmbeddingApp``, ``KMeans/fitSeeded(_:rows:columns:groups:seed:maxIterations:)``
and ``DiffractionFeatures`` reproduce mac4DSTEM's diffraction-embedding numerics as named options.
Streaming, settings and presentation are the caller's.
