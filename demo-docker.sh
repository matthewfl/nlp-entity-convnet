#!/bin/bash

# demo script that uses docker to install deps

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

docker run -it -v $DIR:/project matthewfl/2015-nlp-convnet /project/demo.sh
