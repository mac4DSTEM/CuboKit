# What a consumer changes to import CuboKit 0.1.0

Written 2026-10-15 for mac4DSTEM by the CuboKit session, read at mac4DSTEM's pushed `7f192da9`. mac4DSTEM's own session
makes the switch; it is moving to 0.0.3 today, so this note says what 0.1.0 changes from 0.0.3.

## The short answer

**Nothing breaks.** 0.1.0 is 0.0.3 plus slice 4 (`docs/consumers-0.0.4.md`) plus the packaging that makes the kit stand
on its own; every 0.0.3 call compiles and returns the same numbers. Change the pin from `exact: "0.0.3"` to
`exact: "0.1.0"` and build.

| Since 0.0.3 | What it means for mac4DSTEM |
|---|---|
| Slice 4 (0.0.4): the median and selected virtual diffraction, `get_origin`'s scan mask and fast centre, the Friedel origin, the beamstop mask, the vacuum probe and its kernels | New API to adopt at will; the app's `FriedelOrigin`, `BeamstopMask` and `ProbeKernel` share names with the kit's (`docs/consumers-0.0.4.md`, items 1 to 7) |
| `CuboVirtualError.invalidInput(String)` | An exhaustive `switch` over `CuboVirtualError` needs a `default` |
| A new executable product `cubo-first-look` (`Examples/CuboFirstLook`) | None: an app depends on the library products only; SwiftPM builds the example only when asked |
| README, `docs/parity.md` filled against py4DSTEM's documented API, CI on macOS (`.github/workflows/ci.yml`), `.spi.yml` | None |
| The platform floor | Unchanged: `Package.swift` declares macOS 27. The sources build unchanged at a macOS 15 deployment target and for the iOS 18 Simulator (built, not run); the floor is the owner's decision |

## What 0.1.0 promises

- **Semantic versions from here.** Within 0.1.x, additive changes only: no public symbol removed or changed in meaning,
  no number moved without a registration amendment named in that tag's consumer note. A change that breaks a caller
  comes as 0.2.0.
- **0.2.0 is announced: the generality pass.** Every suite policy becomes a parameter with today's value as its
  documented default, and no suite or app name stays in public docs or comments. Expect call sites to keep compiling (new
  parameters carry defaults); a rename, if any, will be listed in `docs/consumers-0.2.0.md`.
- **The numbers are the registrations'.** Slices 1 to 4 (`docs/registration-slice-*.md`), each held to the float64
  evaluation of py4DSTEM `f050d207`'s definitions on the synthetic pins, planted truth and five real cubes plus the
  bullseye vacuum probe; py4DSTEM's own float32 results are recorded beside them.

## Pinned consumer tests to keep

mac4DSTEM's own consumer tests that pin 0.0.3's numbers stay green at 0.1.0 unchanged; that is the proof the bump moved
no number. If one moves, stop and report it to the kit (the kit loop, suite `ROADMAP.md`).
