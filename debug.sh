WORKDIR=$(pwd)/debug
rm -rf $WORKDIR && mkdir -p $WORKDIR && chmod 777 $WORKDIR

echo starting debug run, workdir: $WORKDIR
python main.py \
    --workdir="$WORKDIR" \
    --mode=local_debug \
    --config=configs/load_config.py:local_debug