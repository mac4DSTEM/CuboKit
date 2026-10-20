#!/bin/zsh
# CuboKit consumer-check negative controls (revert one `public` to `package` in a copy; the build must fail).
#
# The consumer package (tools/consumer-check) compiles only while every symbol it names is `public`. This proves it: for
# each symbol below, ONE declaration is changed from `public` to `package` in a COPY of the kit (never the working tree),
# the consumer is built against the copy, and the build must FAIL with a compiler error that names that symbol. A control
# is caught only when
#   * the unedited copy builds first (the baseline, so a failure is the edit's),
#   * the exact text to change occurs exactly once in its file, and
#   * the build exits non-zero with the expected diagnostic, by kind:
#       access:   the consumer is refused, `error: 'x' is inaccessible due to 'package' protection level`, naming the symbol;
#       library:  the edit is on a type that public signatures use, so the library itself refuses it (`... cannot be declared
#                 public because its parameter uses a package type`, or `... is package and cannot be referenced from a
#                 default argument value`), in a file under Sources/Cubo*;
#       overload: the declaration is one of two overloads, so the consumer's call finds no match: an error in the consumer's
#                 own main.swift (the call is rejected: `missing argument ...` or `extra arguments ...`).
#
#   controls.sh <scratch folder>     (the folder is created; the copy lives in <scratch>/controls/<kit name>)
#
# One warm build folder serves every control: the kit is copied once, one file is changed and restored per control, so
# each build is incremental. Prints one line per control; exit 0 only when every control is caught. Builds use 2 jobs.
set -uo pipefail
here=${0:A:h}
root=${here:h:h}
(( $# == 1 )) || { print -u2 "usage: controls.sh <scratch folder>"; exit 2 }
scratch=${1:A}
[[ $scratch == $root || $scratch == $root/* ]] && { print -u2 "controls.sh: the scratch folder must be outside the kit"; exit 2 }
work=$scratch/controls
copy=$work/${root:t}
logs=$work/logs
mkdir -p $copy $logs
rsync -a --delete --exclude .build --exclude .git --exclude tools/consumer-check/.build $root/ $copy/
consumer=$copy/tools/consumer-check
V=Sources/CuboVirtual

# --disable-dependency-cache: SwiftPM's shared repository cache can print a 'skipping cache' warning on a CI runner,
# which the warning count would take for the kit's own (DiffractoKit 0.4.1's CI).
build() { (cd $consumer && swift build --disable-dependency-cache --jobs 2 -c debug) >$1 2>&1 }

print "baseline: building the unedited copy"
build $logs/baseline.log || { grep "error" $logs/baseline.log | head -10 >&2; print -u2 "controls.sh: the BASELINE does not build ($logs/baseline.log)"; exit 1 }

caught=0
failed=0
restore_file=""
restore() { [[ -n $restore_file ]] && cp $logs/original.swift $restore_file; restore_file="" }
trap restore EXIT INT TERM

# control <kind> <symbol> <file under the kit> <token in the error (access only)> <exact text containing the `public` to change>
control() {
  local kind=$1 symbol=$2 rel=$3 token=$4 old=$5 file=$copy/$3 content parts new log result errors hit
  log=$logs/${symbol//[^A-Za-z0-9_.]/_}.log
  content=$(<$file; print -n x); content=${content%x}
  parts=("${(@ps:$old:)content}")
  if (( ${#parts} != 2 )); then
    print "FAIL  $symbol: the text to change occurs $(( ${#parts} - 1 )) times in $rel, not once"; (( failed++ )); return
  fi
  [[ $old == *public* ]] || { print "FAIL  $symbol: the text to change has no \`public\`"; (( failed++ )); return }
  new=${old/public/package}
  cp $file $logs/original.swift; restore_file=$file
  print -rn -- "${content/"$old"/$new}" >$file
  if build $log; then
    result="FAIL  $symbol: the consumer STILL BUILDS with the declaration package ($rel)"; (( failed++ ))
  else
    errors=$(sed -E $'s/\x1b\\[[0-9;]*m//g' $log | grep "error:" | grep -v "SwiftCompile normal")
    case $kind in
      access)   hit=$(print -r -- $errors | grep -F -- "'$token' is inaccessible due to 'package' protection level" | head -1) ;;
      library)  hit=$(print -r -- $errors | grep -E "Sources/Cubo[A-Za-z]+/.*(uses a package type|is package and cannot be referenced)" | head -1) ;;
      overload) hit=$(print -r -- $errors | grep -F "tools/consumer-check/Sources/consumer-check/main.swift" | head -1) ;;
      *)        hit="" ;;
    esac
    if [[ -n $hit ]]; then
      result="caught $symbol ($kind): ${hit##*error: }"; (( caught++ ))
    else
      result="FAIL  $symbol: the build failed but with no $kind diagnostic (see $log)"; (( failed++ ))
    fi
  fi
  restore
  print -r -- $result
}

# Slice 4.
control access "VirtualDiffraction.median"        $V/SelectedDiffraction.swift "median"            'public static func median(of source'
control access "VirtualDiffraction.pattern"       $V/SelectedDiffraction.swift "pattern"           'public static func pattern(of source'
control library "DiffractionStatistic"             $V/SelectedDiffraction.swift "DiffractionStatistic" 'public enum DiffractionStatistic'
control library "ScanSelection"                    $V/SelectedDiffraction.swift "ScanSelection"     'public enum ScanSelection'
control overload "OriginMeasurement.measure(scanMask:)" $V/OriginMeasurement.swift "measure"         $'public static func measure(of source: some CubeSource, radius: Double? = nil, radiusScale: Double = 1.2,\n                               scanMask: [Bool]?'
control access "OriginMap.replacingNaNWithMean"   $V/OriginMeasurement.swift "replacingNaNWithMean" 'public func replacingNaNWithMean'
control access "FriedelOrigin.measure"            $V/FriedelOrigin.swift       "measure"           'public static func measure(of source'
control access "FriedelOrigin.origin"             $V/FriedelOrigin.swift       "origin"            'public static func origin(of pattern'
control access "BeamstopMask.make"                $V/BeamstopMask.swift        "make"              'public static func make(from pattern'
control access "VacuumProbe.measure"              $V/VacuumProbe.swift         "measure"           'public static func measure(of source'
control access "VacuumProbe.zeroVacuum"           $V/VacuumProbe.swift         "zeroVacuum"        'public static func zeroVacuum'
control access "VacuumProbe.dampVacuum"           $V/VacuumProbe.swift         "dampVacuum"        'public static func dampVacuum'
control access "VacuumProbe.probe"                $V/VacuumProbe.swift         "probe"             'public let probe: DetectorImage'
control access "VacuumProbe.averaged"             $V/VacuumProbe.swift         "averaged"          'public let averaged: DetectorImage'
control access "VacuumProbe.estimate"             $V/VacuumProbe.swift         "estimate"          'public let estimate: ProbeEstimate'
control library "SigmoidProfile"                   $V/VacuumProbe.swift         "SigmoidProfile"    'public enum SigmoidProfile'
control access "ProbeKernel.flat"                 $V/VacuumProbe.swift         "flat"              'public static func flat'
control access "ProbeKernel.gaussian"             $V/VacuumProbe.swift         "gaussian"          'public static func gaussian'
control access "ProbeKernel.sigmoid"              $V/VacuumProbe.swift         "sigmoid"           'public static func sigmoid'
control access "SyntheticProbe.make"              $V/VacuumProbe.swift         "make"              'public static func make(radius'
# 0.5.0 candidate: error normalisation, position correction, the trimmed origin fit, crop and bin, preprocessing and the engines.
P=Sources/CuboPtycho
control access  "OriginFit.fitTrimmed"                     Sources/CuboCore/OriginFit.swift "fitTrimmed"          'public static func fitTrimmed('
control library "TrimmedOriginFit"                         Sources/CuboCore/OriginFit.swift "TrimmedOriginFit"    'public struct TrimmedOriginFit'
control library "PtychographyErrorNormalization"           $P/Ptychography.swift        "PtychographyErrorNormalization" 'public enum PtychographyErrorNormalization'
control library "PtychographyPositionCorrection"           $P/Ptychography.swift        "PtychographyPositionCorrection" 'public struct PtychographyPositionCorrection'
control access  "SingleslicePtychography.Result.positions" $P/Ptychography.swift        "positions"            'public let positions: [PtychographyPosition]'
control library "MixedstateZeroPixelError"                 $P/PtychographyEngines.swift "MixedstateZeroPixelError" 'public enum MixedstateZeroPixelError'
control access  "PtychographyAmplitudes.errorNormalization" $P/PtychographyPreparation.swift "errorNormalization" 'public var errorNormalization'
control access  "PtychographyPreparation.prepare"          $P/PtychographyPreparation.swift "prepare"             'public static func prepare(source'
control access  "MixedstatePtychography.initialProbes"     $P/PtychographyEngines.swift "initialProbes"           'public static func initialProbes('
control access  "MultislicePtychography.propagators"       $P/PtychographyEngines.swift "propagators"             'public static func propagators('
control access  "AxisCrop.enclosing"                       Sources/CuboCore/CubeView.swift "enclosing"            'public static func enclosing('
control overload "ViewedCube"                              Sources/CuboCore/CubeView.swift "ViewedCube"           'public actor ViewedCube'
control library "CubeViewSpecification"                   Sources/CuboCore/CubeView.swift "CubeViewSpecification" 'public struct CubeViewSpecification'
control access  "DPCViews.magnitude"                       Sources/CuboPhase/DPCViews.swift "magnitude"           'public static func magnitude('
control access  "RotationConvention.kit"                   Sources/CuboPhase/DPCViews.swift "kit"                 'public static func kit('
# (ViewedCube: a package-level generic actor is not found by the consumer at all, an error in its own main.swift: kind overload.)
# A slice-1 symbol, to show the earlier checks are guarded too.
control overload "VirtualImaging.image"             $V/VirtualImaging.swift      "image"             'public static func image(of source'

print "controls: $caught caught, $failed not caught"
(( failed == 0 ))
