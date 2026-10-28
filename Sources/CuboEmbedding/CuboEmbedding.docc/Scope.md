# Scope

``PCA``, ``KMeans``, ``NMF`` and ``GaussianMixture`` operate on row-major Double matrices supplied by the caller.
Feature extraction from a cube, settings and progress are not part of this module (the covariance can be streamed with
``PCACovarianceAccumulator``); ``DiffractionFeatures`` only turns one pattern into one vector, and ``Featurization`` starts
from a ``FeatureMatrix`` the caller builds.

Primary behaviour is pinned to scikit-learn 1.4.2: the covariance route to PCA with its `svd_flip` sign convention, and
Lloyd's algorithm from caller-supplied centroids. mac4DSTEM's Float32 PCA, seeded k-means++ and feature binning are
named options pinned to its original source. Not provided: whitening, other solvers, random or k-means++ seeding on the
scikit-learn path, sample weights, prediction of new rows. See `docs/embedding.md`.

``Featurization`` follows py4DSTEM 0.14.19 with scikit-learn 1.9.1 numbers. Not ported: `Featurization.from_braggvectors`
(it needs the Bragg types, and this module does not depend on CuboBragg), `ICA`, `RobustScaler`, `spatial_separation`,
`consensus`, `add_features` and `delete_features`. The `nmf` and `gmm` branches of `get_class_DPs` are buggy upstream and
are not copied; ``Featurization/classPatterns(weights:patterns:pixelCount:threshold:)`` is the weighted class pattern.
NNDSVD starts from an exact SVD where scikit-learn's is randomised.
