#!/bin/bash

mes=$1

git pull
git add .

git commit -m "s｛mes｝"

echo "add and push $mes success!!"