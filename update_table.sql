UPDATE datatable main
SET mode_prime = CASE WHEN t.mode_prime = 'B' THEN 1
    				  WHEN t.mode_prime = 'C' THEN 2
                      WHEN t.mode_prime = 'D' THEN 3
                      WHEN t.mode_prime = 'G' THEN 4
                      WHEN t.mode_prime = 'J' THEN 5
                      WHEN t.mode_prime = 'M' THEN 6
                      WHEN t.mode_prime = 'O' THEN 7
                      WHEN t.mode_prime = 'P' THEN 8
                      WHEN t.mode_prime = 'S' THEN 9
                      WHEN t.mode_prime = 'T' THEN 10
                      WHEN t.mode_prime = 'W' THEN 11
                      ELSE 0 END
FROM (SELECT * FROM tts11.trip) t
WHERE main.gid = t.gid;

UPDATE datatable main
SET age = CASE WHEN p.age < 99 THEN p.age ELSE 0 END,
	n_pers_trips = p.n_pers_tri,
    n_tran_trips = p.n_tran_tri
FROM (SELECT * FROM tts11.pers) p
WHERE main.hhld_num = p.hhld_num AND main.pers_num = p.pers_num;

UPDATE datatable main
SET hhld_pers = h.n_person,
	hhld_veh = h.n_vehicle
FROM (SELECT * FROM tts11.hhld) h
WHERE main.hhld_num = h.hhld_num;

SELECT * FROM datatable ORDER BY gid;
