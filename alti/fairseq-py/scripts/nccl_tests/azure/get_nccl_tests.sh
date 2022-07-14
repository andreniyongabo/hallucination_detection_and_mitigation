#!/bin/bash

azcopy cp --recursive "https://fairacceleastus.blob.core.windows.net/xlmg/nccl_tests/*" $(dirname $0)/bin/
chmod +x $(dirname $0)/bin/*
