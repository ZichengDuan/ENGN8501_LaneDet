pids=`ps xuf | grep "python main_landet.py --train" | awk '{print $2}'`
#pids=`ps xuf | grep "[python] <defunct>" | awk '{print $2}'`
kill -9 $pids
