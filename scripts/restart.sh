#!/usr/bin/env bash
# this script loops forever and restarts jobs that were cancelled due to time limit

while true; do
    jobs=$(grep "DUE TO TIME LIMIT" . -r|grep -v restart|cut -d " " -f5)
    for job in ${jobs}
    do
        rm slurm-${job}.out
        scontrol requeue ${job}
    done
    sleep 60
done
