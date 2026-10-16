# Scope

``PCA`` and ``KMeans`` operate on row-major Double matrices supplied by the caller. Feature extraction from a cube,
streaming, settings and progress are not part of this module; ``DiffractionFeatures`` only turns one pattern into one
vector.

Primary behaviour is pinned to scikit-learn 1.4.2: the covariance route to PCA with its `svd_flip` sign convention, and
Lloyd's algorithm from caller-supplied centroids. mac4DSTEM's Float32 PCA, seeded k-means++ and feature binning are
named options pinned to its original source. Not provided: whitening, other solvers, random or k-means++ seeding on the
scikit-learn path, sample weights, prediction of new rows. See `docs/embedding.md`.
