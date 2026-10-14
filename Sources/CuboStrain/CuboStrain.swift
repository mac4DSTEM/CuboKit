//
//  CuboStrain.swift
//  Placeholder, empty until registered. Planned scope: strain maps from Bragg vectors, py4DSTEM's StrainMap
//  (choose_basis_vectors, fit_basis_vectors, get_strain with its coordinate rotation), the reference lattice from an ROI
//  or the whole scan, against py4DSTEM at the kit's pin and planted lattices with known strain. Its seed is mac4DSTEM's
//  Core/Analysis/StrainMapping.swift and StrainFrame.swift (7448401b). Disk detection, which it consumes, is a module
//  decision for its registration (here or in DiffractoKit's geometry).
//

import CuboCore

/// The module's planned scope, as text, so a consumer can see what is not here yet.
public enum CuboStrain {
    public static let scope = "Strain maps from Bragg vectors (py4DSTEM StrainMap); not registered yet."
}
