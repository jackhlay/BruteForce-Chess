#!/bin/sh
export POD_ORDINAL=${HOSTNAME##*-}
./Crawler -move=$POD_ORDINAL
