#!/bin/sh
ORDINAL=$(echo $HOSTNAME | awk -F'-' '{print $NF}')
exec ./Crawler "move=$ORDINAL"
