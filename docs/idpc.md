# Fourier iDPC

`IDPC.integrate` is the Fourier integration of a DPC field (the method mac4DSTEM used before the kit). It is distinct
from py4DSTEM's iterative DPC, also in CuboPhase. The fixture oracle is a NumPy 2.5.0 float64 transcription of that
algorithm, not a call to py4DSTEM. Synthetic phase derivatives supply a separate
planted truth on the periodic, unregularized case.

The input field must already be in the scan frame. Row components integrate along
scan rows; column components along scan columns. The phase spectrum is
−i(qr·Fr+qc·Fc)/(2π(q²+εq²max)); the zero-frequency coefficient is zero.
Each field component is mean-subtracted before padding. The zero-padded solve
centers the field in the enlarged domain using floor offsets, crops to the original
scan, and subtracts the cropped phase mean. Opposite edges are adjacent for a
periodic solve. Default regularization 1e-4 and default padding factor 2. Both choices are public parameters. Fractional padding is unsupported.

Physical calibration accepts scan sampling in Å per pixel and reciprocal sampling
in Å⁻¹ per detector pixel; the gradient scale is 2π times reciprocal sampling.
No wavelength, voltage, unit-string parser, settings or calibration object is ported.
The result is projected phase in radians up to its fixed zero-mean gauge.

Fixtures (`tools/idpc/pins.py`) cover 7×10 periodic with anisotropic sampling, 8×8 at 2× padding and
regularization 1e-4, and 7×10 at 3× padding with calibrated gradient scale,
anisotropic sampling and regularization 0.03. The odd rectangular padded grid
checks centering, frequency ordering and crop/gauge. All input and output samples
are committed; no private data is included. Double precision throughout, absolute
phase tolerance 1e-12 radians. No bit-exact claim. No microscope validation.

Measured maximum absolute gaps: periodic 7.772e-16 rad, 2× padded 2.221e-16 rad,
3× physically calibrated padded 3.331e-16 rad. Fixture size 16,353 bytes. The tests cover all pins, the periodic
planted phase and the invalid calibration, input and overflow cases.
