A = LOAD '/home/ds503/shared_folder/LinkBookPage.csv' USING PigStorage(',') AS (srno:int, NAME:chararray, occupation:chararray, Ncode:int, educationlevel:chararray);
C = LOAD '/home/ds503/shared_folder/AccessLogs.csv' USING PigStorage(',') AS (AccessId:int, ByWho:int, WhatPage:int, TypeOfAccess:chararray, AccessTime:long);
recent_accesses = FILTER C BY AccessTime >= 870400;
active_users = FOREACH recent_accesses GENERATE ByWho AS srno;
active_users_distinct = DISTINCT active_users;
DUMP active_users_distinct;  -- Ensure this outputs the correct list of distinct active users
outdated_users = JOIN A BY srno LEFT OUTER, active_users_distinct BY srno;
DUMP outdated_users;  -- This should show all users from A with NULLs for non-matching active_users
outdated_users_filtered = FILTER outdated_users BY active_users_distinct::srno IS NULL;
DUMP outdated_users_filtered;  -- This should show only the users who haven't accessed the system in 90 days
outdated_result = FOREACH outdated_users_filtered GENERATE A::srno, A::NAME;
DUMP outdated_result;
-- STORE outdated_result INTO './output_G';

