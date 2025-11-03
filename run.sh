#!/bin/bash

case "$1" in
	r)
		shift
		python clip_viewer/main.py "$@"
		;;
	*)
		echo "invalid option"
		;;
esac
