A = LOAD '/home/ds503/shared_folder/LinkBookPage.csv' USING PigStorage(',') AS (srno:int, NAME:chararray, occupation:chararray, Ncode:int, educationlevel:chararray);
B = LOAD '/home/ds503/shared_folder/AccessLogs.csv' USING PigStorage(',') AS (AccessId:int ,ByWho:int ,WhatPage:int ,TypeOfAccess:chararray ,AccessTime:int);
C = GROUP B BY ByWho;
D = FOREACH C GENERATE group AS ByWho, COUNT(B) AS access_count;
E = ORDER D BY access_count DESC;
F = LIMIT E 10;
G = JOIN F BY ByWho,A BY srno;
H = FOREACH G GENERATE F::ByWho, A::srno, A::NAME, A::occupation;
DUMP H;
STORE H INTO './output_B';
