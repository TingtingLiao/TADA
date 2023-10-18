#!/usr/bin/env bash
PART=$1
STAR=$2
END=$3
CONFIG=$4

CODE_PATH=/mnt/mmtech01/usr/liaotingting/projects/TADA
cd $CODE_PATH

cat $PART | head -n $END | tail -n +$STAR | xargs  -I{} python -m apps.run \
--config $CONFIG \
--text {}
