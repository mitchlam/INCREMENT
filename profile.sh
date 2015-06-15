#!/bin/bash

python -m cProfile -s 'tottime' driver.py $@ > profile/profile.txt 2> profile/profile_err.txt
