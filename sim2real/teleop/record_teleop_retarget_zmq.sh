#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

python record_teleop_retarget_zmq.py "$@"
