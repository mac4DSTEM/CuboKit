// CPU ptychography preprocessing, probe construction and single-slice, mixed-state and multislice reconstruction.

import CuboCore

/// The numerical algorithms currently available in this module.
public enum CuboPtycho {
    /// Preprocessing, probe construction and the CPU single-slice, mixed-state and multislice engines.
    public static let scope = "Preprocessing (amplitudes, scan positions, object canvas, starting probe), probe construction and CPU single-slice, mixed-state and multislice ptychography; full batches; probe-centre, probe-amplitude, aperture and support constraints; py4DSTEM's error normalisation; position correction (single-slice)."
}
