#!/bin/bash

# A wrapper around gather_azhpc_vm_diagnostics.sh that only prints the path to the .tar.gz diagnostics

# You are not running the latest release of this tool. Switch to latest version? [y/N] N
# Please confirm that you understand [y/N] Y
printf 'N\nY' | sudo bash /opt/azurehpc/diagnostics/gather_azhpc_vm_diagnostics.sh | grep "^#.*.tar.gz #$" | sed "s/\s*#\s*//g"
