#!/usr/bin/env bash
# The container's start: the model (a catalog key, see `./zero-launcher.sh --list`, or a path to a .gguf in a mounted
# folder) downloaded into the Hugging Face cache if it is a key and not there yet, then served on all interfaces.
#   MODEL       catalog key or .gguf path (default 4b)
#   PORT        the server's port inside the container (default 8090)
#   GPUS        GPUs to split the model over (default 1)
#   SERVER_ARGS more serverz options, e.g. "--ctx 32768"
#   OPTIONS     the speed and accuracy options (docs/options.md; `./zero-launcher.sh --options` lists them), a comma or
#               space list: an id turns one on ("copy_draft"), id=off turns it off ("xcpu=off"), id=value gives one its
#               value ("budget=12"). Exactly these, from the defaults, at every start: none saved in the data volume
set -eu
cd /opt/inferzero
./zero-launcher.sh --check || true
opts=()
OPTIONS=${OPTIONS:-}
for o in ${OPTIONS//,/ }; do
  case "$o" in
    *=off) opts+=(--no-opt "${o%=off}");;
    *=default) opts+=(--default-opt "${o%=default}");;
    *) opts+=(--opt "$o");;
  esac
done
# (--host 0.0.0.0: reachable through the published port; compose.yaml publishes it on 127.0.0.1 only, since the
#  server has no password of its own. --yes: the download without a question.)
# shellcheck disable=SC2086
exec ./zero-launcher.sh --model "${MODEL:-4b}" --host 0.0.0.0 --port "${PORT:-8090}" --gpus "${GPUS:-1}" \
  --no-browser --yes --no-saved-options ${opts[@]+"${opts[@]}"} -- ${SERVER_ARGS:-}
