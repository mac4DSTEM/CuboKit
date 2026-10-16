/// Symmetric eigendecomposition (LAPACK dsyevd, upper triangle). `a` is n×n, column-major, overwritten with the
/// eigenvectors (column j belongs to `w[j]`); `w` receives the eigenvalues in ascending order. Returns LAPACK info.
int cubo_embedding_symmetric_eigen(int n, double *a, double *w);

/// c (n×n, row-major, upper triangle only) = alpha · aᵀa + beta · c, for a of `k × n`, row-major (BLAS dsyrk).
void cubo_embedding_syrk_upper(int n, int k, double alpha, const double *a, double beta, double *c);
