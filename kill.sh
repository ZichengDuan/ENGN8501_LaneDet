<<<<<<< HEAD
ds=`ps xuf | grep "python main_landet.py" | awk '{print $2}'`
=======
pids=`ps xuf | grep "python main_landet.py --train" | awk '{print $2}'`
>>>>>>> 377d819135e9595aba8bc3c36c8fcb59cdc0f1d2
#pids=`ps xuf | grep "[python] <defunct>" | awk '{print $2}'`
kill -9 $pids
