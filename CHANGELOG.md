# Changelog

## Unreleased

- Add CuboPtycho CPU probe construction (defocus, two-fold astigmatism, C30) and
  full-batch single-slice complex-object gradient descent from preprocessed amplitudes,
  with fixed positions, optional probe updates and pure-phase object constraints.
  Pinned py4DSTEM 0.14.19 fixtures cover probe pixels and object/probe after three iterations.

Migration: additive API; preprocessing and other reconstruction methods remain unsupported.
