#!/bin/bash

mes=$1

git pull
git add .

git commit -m "${mes}"

git push -u origin main

echo "add and push $mes success!!"