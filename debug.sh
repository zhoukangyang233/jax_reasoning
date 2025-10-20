now=`date '+%Y%m%d_%H%M%S'`
WORKDIR=$(pwd)/debug/$now
rm -rf $WORKDIR && mkdir -p $WORKDIR && chmod 777 $WORKDIR

echo starting debug run, workdir: $WORKDIR
sleep 1
python main.py \
    --workdir="$WORKDIR" \
    --mode=local_debug \
    --config=configs/load_config.py:local_debug