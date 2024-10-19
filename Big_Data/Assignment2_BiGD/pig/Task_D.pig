A = LOAD '/home/ds503/shared_folder/LinkBookPage.csv' USING PigStorage(',') AS (srno:int, NAME:chararray, occupation:chararray, Ncode:int, educationlevel:chararray);
B = LOAD '/home/ds503/shared_folder/Associates.csv' USING PigStorage(',') AS (ID1:int, ID2:int, DateofRelation:int, Desc:chararray);
C = LOAD '/home/ds503/shared_folder/AccessLogs.csv' USING PigStorage(',') AS (AccessId:int ,ByWho:int ,WhatPage:int ,TypeOfAccess:chararray ,AccessTime:int);
D = FOREACH B GENERATE ID1 AS USER;
E = FOREACH B GENERATE ID2 AS USER;
F = UNION D, E;
G = GROUP F BY USER;
H = FOREACH G GENERATE group AS USER, COUNT(F) AS WEED;
I = JOIN A BY srno LEFT OUTER, H BY USER;
J = FOREACH I GENERATE A::NAME,
	(CASE
	    WHEN H::WEED IS NULL THEN 0
	    ELSE H::WEED
	END) AS WEEED; 
DUMP J;
STORE J INTO './output_D';
