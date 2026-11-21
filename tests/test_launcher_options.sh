#!/usr/bin/env bash
# The Linux launcher's speed and accuracy options (QUEUE O3): --options, --opt / --no-opt / --default-opt and their
# checks, options.json in both of zero-server's forms, the choices on the command line, in the environment and in
# the systemd unit. In a temporary folder (ARC_DATA_DIR), no GPU, no network. Run by CI.
#   bash tests/test_launcher_options.sh
ROOTW=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
T=$(mktemp -d)
export ARC_DATA_DIR="$T"
cd "$ROOTW" || exit 1
L=./zero-launcher.sh
pass=0; fail=0
ok() { if [ "$1" -eq 0 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL: $2"; fi; }

$L --options > "$T/list.txt"; ok $? "--options runs"
grep -q "copy_draft" "$T/list.txt"; ok $? "--options lists copy_draft"
grep -q "now: default" "$T/list.txt"; ok $? "nothing chosen yet"

$L --opt copy_draft --no-opt xcpu --opt budget=12 --options > "$T/list2.txt"; ok $? "--opt / --no-opt with --options"
grep -q '"copy_draft":  "on"' "$T/options.json"; ok $? "json copy_draft on"
grep -q '"xcpu":  "off"' "$T/options.json"; ok $? "json xcpu off"
grep -q '"budget":  "12"' "$T/options.json"; ok $? "json budget 12"

$L --opt budget --options > /dev/null 2>&1; [ $? -eq 2 ]; ok $? "budget without a value refused"
$L --opt split_layer --options > /dev/null 2>&1; ok $? "split_layer on without a value accepted"
$L --default-opt split_layer --options > /dev/null; ok $? "split_layer back to the default"
$L --opt nosuch --options > /dev/null 2>&1; [ $? -eq 2 ]; ok $? "an unknown id refused"
$L --opt 'budget=1"2' --options > /dev/null 2>&1; [ $? -eq 2 ]; ok $? "a quote refused"
grep -q '"budget":  "12"' "$T/options.json"; ok $? "a refused change leaves the file"

$L --default-opt budget --options > /dev/null; ok $? "--default-opt"
grep -q budget "$T/options.json"; [ $? -ne 0 ]; ok $? "budget gone after --default-opt"

# --no-saved-options (Docker's OPTIONS): exactly the ones given, from the defaults, nothing written
printf '{"xcpu":"off"}' > "$T/options.json"
$L --no-saved-options --opt copy_draft --options > "$T/list3.txt"; ok $? "--no-saved-options runs"
grep -A0 "copy_draft" "$T/list3.txt" | grep -q "now: on"; ok $? "the given option is on"
grep "xcpu " "$T/list3.txt" | grep -q "now: default"; ok $? "the saved choice is not used"
[ "$(cat "$T/options.json")" = '{"xcpu":"off"}' ]; ok $? "options.json left as it was"

# a file in zero-server's single-line form (PowerShell 7's -Compress, or written by hand)
printf '{"batch":"0","gemv_gu":"on"}' > "$T/options.json"
# the effect: source the launcher's functions
source "$L"
opt_load
[ "${OPTC[batch]}" = 0 ] && [ "${OPTC[gemv_gu]}" = on ]; ok $? "a one-line json read"
PORT=8090; HOST=""; GPUS=0; EXTRA=(); SERVERZ=/bin/serverz
server_cmd /m.gguf 1
printf '%s\n' "${OPT_ENV[@]}" | grep -qx "ARC_BATCH=0"; ok $? "batch off -> ARC_BATCH=0"
printf '%s\n' "${OPT_ENV[@]}" | grep -qx "ARC_GEMV_GU=1"; ok $? "gemv_gu on -> ARC_GEMV_GU=1"
OPTC=([copy_draft]=on [budget]=12 [fn_precise]=off)
server_cmd /m.gguf 0
c="${CMD[*]}"
case "$c" in *"--copy-draft"*) r=0;; *) r=1;; esac; ok $r "copy_draft flag on the command line"
case "$c" in *"--budget 12"*) r=0;; *) r=1;; esac; ok $r "budget 12 on the command line"
case "$c" in *"--precise"*) r=1;; *) r=0;; esac; ok $r "a flag chosen off is left out"
OPTC=([split_layer]=on)
server_cmd /m.gguf 0
case "${CMD[*]}" in *"--split layer"*) r=0;; *) r=1;; esac; ok $r "split_layer on -> --split layer (a word in on: its value, not a placeholder)"
[ ${#OPT_ENV[@]} -eq 0 ]; ok $? "no env from flags"

# the unit carries the env options
OPTC=([batch]=on)
SERVICE_NAME=inferzero-opt-test; REPLACE_SERVICE=0; XDG_CONFIG_HOME="$T/cfg"; ROOT="$ROOTW"
service_refused() { :; }
service_unit() { printf '%s' "$T/cfg/systemd/user/inferzero-opt-test.service"; }
server_cmd /m.gguf 0
install_service > /dev/null 2>&1
grep -q '^Environment="ARC_BATCH=64"' "$T/cfg/systemd/user/inferzero-opt-test.service"; ok $? "the unit has Environment=ARC_BATCH=64"

rm -rf "$T"
echo "test_launcher_options: $pass passed, $fail failed"
[ $fail -eq 0 ]
