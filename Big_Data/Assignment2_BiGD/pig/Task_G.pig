A = LOAD '/home/ds503/shared_folder/LinkBookPage.csv' USING PigStorage(',') AS (srno:int, NAME:chararray, occupation:chararray, Ncode:int, educationlevel:chararray);
B = LOAD '/home/ds503/shared_folder/Associates.csv' USING PigStorage(',') AS (ID1:int, ID2:int, DateofRelation:int, Desc:chararray);
C = LOAD '/home/ds503/shared_folder/AccessLogs.csv' USING PigStorage(',') AS (AccessId:int, ByWho:int, WhatPage:int, TypeOfAccess:chararray, AccessTime:int);

N = GROUP C BY ByWho;
O = FOREACH N GENERATE group AS USER, MAX(C.AccessTime) AS LAST_ACCESS;

current_time = FOREACH (GROUP C ALL) GENERATE MAX(C.AccessTime) AS MAX_TIME;


P = CROSS O, current_time;
Q = FOREACH P GENERATE 
    O::USER AS USER,
    O::LAST_ACCESS AS LAST_ACCESS,
    current_time::MAX_TIME AS CURRENT_TIME,
    ((current_time::MAX_TIME - O::LAST_ACCESS + 1000000) % 1000000) AS TIME_DIFF;


R = FILTER Q BY TIME_DIFF > 129600 OR LAST_ACCESS IS NULL;


S = JOIN R BY USER RIGHT OUTER, A BY srno;
result_g = FOREACH S GENERATE 
    A::NAME,
    (CASE
        WHEN R::LAST_ACCESS IS NULL THEN 'Never accessed'
        ELSE CONCAT('Last accessed ', ToString(R::LAST_ACCESS), ' (', ToString(R::TIME_DIFF), ' minutes ago)')
    END) AS ACCESS_INFO;

STORE result_g INTO 'Output_Task_G' USING PigStorage(',');
