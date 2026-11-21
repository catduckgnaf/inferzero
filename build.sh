#!/usr/bin/env bash
# Builds inferzero's Linux programs into bin/ with g++ (docs/linux-port.md): the CLI (runz), the headless server
# (serverz), the multi-GPU link probe (mgpu_probe), the f32 CPU reference the KL checks compare with (ref_forward:
# bench/website_bench.sh, runz --kl-ref) and the GPU's peak rates (gpu_bench). The Windows build is build.ps1; the
# kernels are compiled at run time by Intel's driver, as on Windows.
#   ./build.sh                 every program
#   ./build.sh runz serverz    only these (./build.sh xw_peak: the XMX GEMM tile probe, bench/xw_peak.cpp)
# (ref_forward is built without -ffast-math, as the Windows build keeps it off /fp:fast: it is the reference)
# Needs g++ 13 or newer (C++20) and an x86-64 CPU; with AVX2 / FMA / F16C (as the Windows build) or, without them,
# a build for this CPU alone (-march=native, below). Nothing else at
# build time: Level Zero and OpenCL are loaded at run time (libze_loader.so.1, libOpenCL.so.1).
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
declare -A SRC=([runz]=zero/runz.cpp [serverz]=zero/serverz.cpp [mgpu_probe]=zero/mgpu_probe.cpp
                [ref_forward]=tools/ref_forward.cpp [gpu_bench]=bench/gpu_bench.cpp [xw_peak]=bench/xw_peak.cpp)
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(runz serverz mgpu_probe ref_forward gpu_bench)
for t in "${targets[@]}"; do
  [ -n "${SRC[$t]:-}" ] || { echo "build.sh: unknown program '$t' (runz, serverz, mgpu_probe, ref_forward, gpu_bench, xw_peak)" >&2; exit 2; }
done
command -v "$CXX" >/dev/null || { echo "build.sh: $CXX not found (sudo apt install g++)" >&2; exit 1; }
mkdir -p bin
# (-rdynamic: the crash report's function names; -include: the CPU check and the log, as build.ps1's /FI)
# (the CPU: AVX2 / FMA / F16C as the release packages; a CPU without them (Ivy Bridge, an older Xeon with an A770)
#  gets -march=native, so the programs use what it has and src/cpu_check.h checks just that; ThatGuy-Bob, issue
#  #50. INFERZERO_MARCH=<arch> sets it, e.g. INFERZERO_MARCH=ivybridge for another machine)
ARCH=(-mavx2 -mfma -mf16c)
if [ -n "${INFERZERO_MARCH:-}" ]; then
  ARCH=(-march="$INFERZERO_MARCH")
elif [ -r /proc/cpuinfo ] && ! { grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo; }; then
  ARCH=(-march=native)
  echo "build.sh: this CPU has no AVX2 / FMA: built for it alone (-march=native), not for other machines"
fi
FLAGS=(-std=c++20 -O2 "${ARCH[@]}" -fopenmp -rdynamic -include src/cpu_check.h -include src/log.h -I third_party -pthread)
# (-ldl: dlopen lives in libdl before glibc 2.34, e.g. Debian 11 / RHEL 8; harmless on newer systems)
LIBS=(-ldl)
# (a package's bin/PREBUILT.txt names its release; a build here is no longer that release)
rm -f bin/PREBUILT.txt
pids=()
for t in "${targets[@]}"; do
  echo "build.sh: $t"
  "$CXX" "${FLAGS[@]}" "${SRC[$t]}" -o "bin/$t" "${LIBS[@]}" ${LDFLAGS:-} &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "build.sh: a build failed" >&2; exit 1; }
echo "build.sh: done (${targets[*]} in bin/)"
