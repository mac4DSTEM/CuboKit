# Contributing to CuboKit

Thanks for your interest. Bug reports, questions and pull requests are welcome.

## Reporting issues

Open an issue and say what you ran, what you expected and what you got. For a wrong number, a small cube (or the code
that makes one), the call you made and, if you have it, py4DSTEM's answer help most. Please say which Mac, macOS and
Swift version you used and whether you used `.cpu` or `.metal`.

## Building and testing

CuboKit needs macOS 15 or later and a Swift 6.2 toolchain.

```sh
swift build
swift test -c release
```

The tests use Swift Testing (`import Testing`, `@Test`, `#expect`). They need nothing outside the package: the small
reference fixtures are committed. Tests that need large real cubes skip themselves when the data is absent.

## Numbers are checked against py4DSTEM

Results are compared with fixtures produced by py4DSTEM at a pinned commit (`f050d207`, 0.14.19), or with a closed form
or a planted cube. A test's tolerance is the one the Python package's own tests use for that function; a result outside it is a bug. A change that moves a fixture says why in the pull request and in the CHANGELOG. Regenerate the fixture in its own commit.

## Pull requests

- Keep a change focused, with a test that fails before it and passes after.
- Public API only: a type a user needs must be `public`, and nothing should name an application.
- Say in the pull request what you ran. Please do not edit generated fixtures by hand.
- By contributing you agree that your contribution is licensed under GPL-3.0, like the rest of the package.
