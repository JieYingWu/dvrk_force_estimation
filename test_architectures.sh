#!/bin/sh

window=( 1 5 10 10 10 10 20 50 100)
skip=( 10 10 2 5 10 100 100 10 10)

for i in {1..9}
do
   echo "Running for window size ${window[i]} and skip ${skip[i]}"
   python test_indirect_no_sensor.py free_space ${window[i]} ${skip[i]}
done