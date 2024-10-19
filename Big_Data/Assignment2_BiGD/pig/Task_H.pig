A = LOAD '/home/ds503/shared_folder/LinkBookPage.csv' USING PigStorage(',') AS (srno:int, NAME:chararray, occupation:chararray, Ncode:int, educationlevel:chararray);
B = LOAD '/home/ds503/shared_folder/Associates.csv' USING PigStorage(',') AS (ID1:int, ID2:int, DateofRelation:int, Desc:chararray);
C = LOAD '/home/ds503/shared_folder/AccessLogs.csv' USING PigStorage(',') AS (AccessId:int, ByWho:int, WhatPage:int, TypeOfAccess:chararray, AccessTime:int);

R = FOREACH B GENERATE ID1 AS USER, ID2 AS FRIEND;
S = FOREACH B GENERATE ID2 AS USER, ID1 AS FRIEND;
T = UNION R, S;


U = FOREACH C GENERATE ByWho AS USER, WhatPage AS ACCESSED_PAGE;


V = JOIN T BY (USER, FRIEND) LEFT OUTER, U BY (USER, ACCESSED_PAGE);
W = FILTER V BY U::USER IS NULL;


X = GROUP W BY T::USER;
Y = FOREACH X GENERATE group AS USER;


Z = JOIN Y BY USER, A BY srno;
result_h = FOREACH Z GENERATE A::NAME;

STORE result_h INTO './output_task_H' USING PigStorage(',');
