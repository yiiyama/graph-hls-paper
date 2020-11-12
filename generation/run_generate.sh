#!/bin/bash

## EDIT BELOW
OUTBASE=/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/generated2
## EDIT ABOVE

PART=$1
EMIN=$2
EMAX=$3
NGEN=$4
JOBID=$5
SEEDBASE=$6

[ $SEEDBASE ] || SEEDBASE=123456

THISDIR=$(cd $(dirname $(readlink -f $0)); pwd)

source $THISDIR/env.sh

SEED=$(($SEEDBASE+$JOBID))

if [ $PART = "pileup" ]
then
  $THISDIR/generate -f events.root -p - -u -n $NGEN -s $SEED
else
  $THISDIR/generate -f events.root -e $EMIN $EMAX -p $PART -n $NGEN -s $SEED
fi

OUTDIR=$OUTBASE/${PART}_${EMIN}_${EMAX}/$SEEDBASE

mkdir -p $OUTDIR

cp events.root $OUTDIR/events_$JOBID.root
