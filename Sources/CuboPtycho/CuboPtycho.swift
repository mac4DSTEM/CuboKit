//
//  CuboPtycho.swift
//  Placeholder, empty until registered. Planned scope: iterative ptychography, py4DSTEM's SingleslicePtychography first
//  (then mixed-state and multislice), and tilt-corrected bright field (parallax) with its aberration fit, Metal first
//  with a CPU reference and a registered float32-against-float64 tolerance. Its seeds are mac4DSTEM's
//  Core/Analysis/SingleslicePtychography.swift, PtychographyPreparation.swift and the Parallax*.swift files (7448401b);
//  SimulatoKit's probes and multislice are its truth source.
//

import CuboCore

/// The module's planned scope, as text, so a consumer can see what is not here yet.
public enum CuboPtycho {
    public static let scope = "Ptychography and tilt-corrected bright field (py4DSTEM phase retrieval); not registered yet."
}
