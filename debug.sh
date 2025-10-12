WORKDIR=$(pwd)/debug
rm -rf $WORKDIR && mkdir -p $WORKDIR && chmod 777 $WORKDIR

echo starting debug run, workdir: $WORKDIR
JAX_PLATFORMS=cpu python main.py \  
    --workdir=$WORKDIR \
    --mode=local_debug \
    --config=configs/debug_config.yml