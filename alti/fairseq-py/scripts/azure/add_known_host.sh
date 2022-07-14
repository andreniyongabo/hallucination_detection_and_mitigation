#!/bin/bash

HOSTNAME=$1
HOST_IP=$(getent hosts $HOSTNAME | awk '{ print $1 }')

ssh-keyscan -t ecdsa-sha2-nistp256 -H $HOSTNAME 2> /dev/null >> ~/.ssh/known_hosts
ssh-keyscan -t ecdsa-sha2-nistp256 -H $HOST_IP 2> /dev/null >> ~/.ssh/known_hosts
