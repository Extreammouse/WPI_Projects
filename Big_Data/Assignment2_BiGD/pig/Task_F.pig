A = LOAD '/home/ds503/shared_folder/LinkBookPage.csv' USING PigStorage(',') AS (srno:int, NAME:chararray, occupation:chararray, Ncode:int, educationlevel:chararray);
B = LOAD '/home/ds503/shared_folder/Associates.csv' USING PigStorage(',') AS (ID1:int, ID2:int, DateofRelation:int, Desc:chararray);
C = LOAD '/home/ds503/shared_folder/AccessLogs.csv' USING PigStorage(',') AS (AccessId:int, ByWho:int, WhatPage:int, TypeOfAccess:chararray, AccessTime:int);


D = FOREACH B GENERATE ID1 AS USER;
E = FOREACH B GENERATE ID2 AS USER;
F = UNION D, E;
G = GROUP F BY USER;
H = FOREACH G GENERATE group AS USER, COUNT(F) AS RELATIONSHIP_COUNT;

I = GROUP H ALL;
J = FOREACH I GENERATE AVG(H.RELATIONSHIP_COUNT) AS AVG_RELATIONSHIPS;

K = CROSS H, J;
L = FILTER K BY H::RELATIONSHIP_COUNT > J::AVG_RELATIONSHIPS;

M = JOIN L BY H::USER, A BY srno;
result_f = FOREACH M GENERATE A::NAME, L::H::RELATIONSHIP_COUNT;

STORE result_f INTO './output_taskf' USING PigStorage(',');
