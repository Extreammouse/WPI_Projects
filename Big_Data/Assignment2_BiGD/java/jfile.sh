#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <subdirectory-name>"
    exit 1
fi

BASE_PATH="/user/ds503/project2/outputs"
SUBDIR=$1
OUTPUT_PATH="${BASE_PATH}/${SUBDIR}"
mkdir -p "project2"
javac -classpath $(hadoop classpath) -d . KMeansClustering.java
jar cf kmeans.jar project2/*.class
jar tf kmeans.jar
#hadoop jar kmeans.jar project2.KMeansClustering /user/ds503/project2/input/large_dataset.csv $OUTPUT_PATH > ./output_converge_old.txt
hadoop jar kmeans.jar project2.KMeansClustering /user/ds503/project2/input/df_reduced.csv $OUTPUT_PATH > ./output_newdataset_convergance.txt
#~/hadoop/bin/hdfs dfs -cat $OUTPUT_PATH/part-r-00000 >> Output.txt
rm -rf project2/*
