A = LOAD '/home/ds503/shared_folder/LinkBookPage.csv' USING PigStorage(',') AS (srno:int, NAME:chararray, occupation:chararray, Ncode:int, educationlevel:chararray);
B = LOAD '/home/ds503/shared_folder/AccessLogs.csv' USING PigStorage(',') AS (AccessId:int ,ByWho:int ,WhatPage:int ,TypeOfAccess:chararray ,AccessTime:int);
C = FILTER A BY educationlevel == 'Masters';
D = FOREACH C GENERATE srno, NAME, occupation;
DUMP D;
STORE D INTO './output_C';
