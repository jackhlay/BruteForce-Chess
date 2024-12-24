#!/bin/sh
POD_ORDINAL=${HOSTNAME##*-}
exec ./Crawler -move=$POD_ORDINAL
