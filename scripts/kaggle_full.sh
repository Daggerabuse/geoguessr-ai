#!/usr/bin/env bash
# Phase 1 full run: all 98 train zips, 11 h deadline, archive to
# Daggerabuse/osv5m-ft-archive. See kaggle_phase1.sh.
exec bash "$(dirname "$0")/kaggle_phase1.sh" full
