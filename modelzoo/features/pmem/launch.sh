#!/bin/bash

set -x

CLUSTER="--ps_hosts=127.0.0.1:8898 --worker_hosts=127.0.0.1:8860"
# launch ps
python ./benchmark.py $CLUSTER --job_name=ps  --task_index=0 $@ > bench-ps.log &
pid_ps=$!

measure_rss() {
# record rss usage
logfile=./ps-mem-rss-trace.log
echo "start recording rss " > $logfile
start=$(date +%s)
# get the process' memory usage and run until `ps` fails which it will do when
# the pid cannot be found any longer
while mem=$(ps -o rss= -p "$pid_ps"); do
    time=$(date +%s)
    # print the time since starting the program followed by its memory usage
    printf "%d %s\n" $((time-start)) "$mem" >> $logfile
    # sleep for a tenth of a second
    sleep .1
done
printf "Find the log at %s\n" "$logfile"
}
measure_rss &>/dev/null &
# launch worker
python ./benchmark.py $CLUSTER --job_name=worker --task_index=0 $@ 2>&1 | tee bench-worker.log
# clean procs
ps aux | grep benchmark.py | awk '{print $2}' | xargs kill -9
wait



