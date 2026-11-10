#!/bin/bash
# Detached launch (survives SSH drops). Run ONCE; refuses if a chain is already alive. Never wrap in a retry loop.
B=/root/a38
if [ -f $B/chain.pid ] && kill -0 "$(cat $B/chain.pid)" 2>/dev/null; then echo "chain already running pid $(cat $B/chain.pid)"; exit 1; fi
setsid nohup bash $B/bundle/scripts/run_a38.sh > $B/chain.nohup 2>&1 < /dev/null &
echo $! > $B/chain.pid
echo "launched pid $(cat $B/chain.pid)"
