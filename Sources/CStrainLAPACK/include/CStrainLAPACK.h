/// Minimum-norm least squares, column-major. Returns LAPACK info.
int cubo_strain_least_squares(int rows, int columns, double *design, double *rhs, double rcond);
