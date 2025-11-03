#!/bin/bash

case "$1" in
	r)
		shift
		python clip_viewer/main.py "$@"
		;;
	v)
		shift
		python clip_viewer/visu_distance.py "$@"
		;;
	*)
		echo "invalid option"
		;;
esac
