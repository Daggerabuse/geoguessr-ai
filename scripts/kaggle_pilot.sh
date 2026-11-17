#!/usr/bin/env bash
# Phase 1 pilot: 3 train zips, 40 min, archive to the *-pilot repo. See kaggle_phase1.sh.
export PYTHONUNBUFFERED=1  # live Kaggle logs
exec bash "$(dirname "$0")/kaggle_phase1.sh" pilot
