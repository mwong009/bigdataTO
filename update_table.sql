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
        ELSE 0 END,
	trip_km = CASE WHEN t.trip_km < 999 THEN t.trip_km
        ELSE 0 END,
    car_pool = CASE WHEN t.car_pool < 99 THEN t.car_pool
        ELSE 0 END,
    hwy407 = CASE WHEN t.hwy407 = 'Y' THEN 1
        WHEN t.hwy407 = 'N' THEN -1
        ELSE 0 END,
    trip_type = CASE WHEN t.trip_purp < 4 THEN 1
        WHEN t.trip_purp = 4 THEN -1
        ELSE 0 END,
    trip_orig_reg = CASE WHEN t.region_ori <= 6 THEN t.region_ori
        ELSE 0 END,
    trip_dest_reg = CASE WHEN t.region_des <= 6 THEN t.region_des
        ELSE 0 END,
    trip_orig_pd = CASE WHEN t.pd_orig = 1 THEN 1
        WHEN t.pd_orig = 2 OR t.pd_orig = 4 OR t.pd_orig = 6 THEN 2
        WHEN t.pd_orig = 3 THEN 3
        WHEN t.pd_orig = 5 THEN 4
        WHEN t.pd_orig = 10 OR t.pd_orig = 11 OR t.pd_orig = 12 THEN 5
        WHEN t.pd_orig = 7 OR t.pd_orig = 8 OR t.pd_orig = 9 THEN 6
        WHEN t.pd_orig > 12 AND t.pd_orig <= 16 THEN 7
        WHEN t.pd_orig > 16 AND t.pd_orig <= 46 THEN 8
        ELSE 0 END,
    trip_dest_pd = CASE WHEN t.pd_dest = 1 THEN 1
        WHEN t.pd_dest = 2 OR t.pd_dest = 4 OR t.pd_dest = 6 THEN 2
        WHEN t.pd_dest = 3 THEN 3
        WHEN t.pd_dest = 5 THEN 4
        WHEN t.pd_dest = 10 OR t.pd_dest = 11 OR t.pd_dest = 12 THEN 5
        WHEN t.pd_dest = 7 OR t.pd_dest = 8 OR t.pd_dest = 9 THEN 6
        WHEN t.pd_dest > 12 AND t.pd_dest <= 16 THEN 7
        WHEN t.pd_dest > 16 AND t.pd_dest <= 46 THEN 8
        ELSE 0 END
FROM (SELECT * FROM tts11.trip) t
WHERE main.gid = t.gid
    AND (main.mode_prime IS NULL OR main.trip_km IS NULL OR
        main.car_pool IS NULL OR main.hwy407 IS NULL OR
        main.trip_type IS NULL OR main.trip_orig_reg IS NULL OR
        main.trip_dest_reg IS NULL OR main.trip_orig_pd IS NULL OR
        main.trip_dest_pd IS NULL);

UPDATE datatable main
SET age = CASE WHEN p.age < 99 THEN p.age ELSE 0 END,
	n_pers_trips = CASE WHEN p.n_pers_tri < 99 THEN p.n_pers_tri
        ELSE 0 END,
    n_tran_trips = CASE WHEN p.n_tran_tri < 99 THEN p.n_tran_tri
        ELSE 0 END,
    sex = CASE WHEN p.sex = 'M' THEN -1
        WHEN p.sex = 'F' THEN 1
        ELSE 0 END,
    driver_lic = CASE WHEN p.driver_lic = 'N' THEN -1
        WHEN p.driver_lic = 'Y' THEN 1
        ELSE 0 END,
    pass_ttc = CASE WHEN p.tran_pass = 'M' OR p.tran_pass = 'C' THEN 1
        WHEN p.tran_pass = '9' OR p.tran_pass = 'N' THEN 0
        ELSE -1 END,
    pass_go = CASE WHEN p.tran_pass = 'G' OR p.tran_pass = 'C' THEN 1
        WHEN p.tran_pass = '9' OR p.tran_pass = 'N' THEN 0
        ELSE -1 END,
    pass_oth = CASE WHEN p.tran_pass = 'O' THEN 1
        WHEN p.tran_pass = '9' OR p.tran_pass = 'N' THEN 0
        ELSE -1 END,
    emp_ft = CASE WHEN p.emp_stat = 'F' OR p.emp_stat = 'H' THEN 1
        WHEN p.emp_stat = 'O' OR p.emp_stat = '9' THEN 0
        ELSE -1 END,
    emp_workhome = CASE WHEN p.emp_stat = 'H' OR p.emp_stat = 'J' THEN 1
        WHEN p.emp_stat = 'O' OR p.emp_stat = '9' THEN 0
        ELSE -1 END,
    student = CASE WHEN p.stu_stat = 'P' OR p.stu_stat = 'S' THEN 1
        WHEN p.stu_stat = 'O' THEN -1
        ELSE 0 END,
    free_park = CASE WHEN p.free_park = 'Y' THEN 1
        WHEN p.free_park = 'N' THEN -1
        ELSE 0 END,
    occupation = CASE WHEN p.occupation = 'G' THEN 1
        WHEN p.occupation = 'M' THEN 2
        WHEN p.occupation = 'P' THEN 3
        WHEN p.occupation = 'S' THEN 4
        WHEN p.occupation = 'O' THEN 5
        ELSE 0 END,
    emp_region = CASE WHEN p.region_emp <= 6 THEN p.region_emp
        ELSE 0 END,
    sch_region = CASE WHEN p.region_sch <= 6 THEN p.region_sch
        ELSE 0 END,
    emp_pd = CASE WHEN p.pd_emp = 1 THEN 1
        WHEN p.pd_emp = 2 OR p.pd_emp = 4 OR p.pd_emp = 6 THEN 2
        WHEN p.pd_emp = 3 THEN 3
        WHEN p.pd_emp = 5 THEN 4
        WHEN p.pd_emp = 10 OR p.pd_emp = 11 OR p.pd_emp = 12 THEN 5
        WHEN p.pd_emp = 7 OR p.pd_emp = 8 OR p.pd_emp = 9 THEN 6
        WHEN p.pd_emp > 12 AND p.pd_emp <= 16 THEN 7
        WHEN p.pd_emp > 16 AND p.pd_emp <= 46 THEN 8
        ELSE 0 END,
    sch_pd = CASE WHEN p.pd_sch = 1 THEN 1
        WHEN p.pd_sch = 2 OR p.pd_sch = 4 OR p.pd_sch = 6 THEN 2
        WHEN p.pd_sch = 3 THEN 3
        WHEN p.pd_sch = 5 THEN 4
        WHEN p.pd_sch = 10 OR p.pd_sch = 11 OR p.pd_sch = 12 THEN 5
        WHEN p.pd_sch = 7 OR p.pd_sch = 8 OR p.pd_sch = 9 THEN 6
        WHEN p.pd_sch > 12 AND p.pd_sch <= 16 THEN 7
        WHEN p.pd_sch > 16 AND p.pd_sch <= 46 THEN 8
        ELSE 0 END

FROM (SELECT * FROM tts11.pers) p
WHERE main.hhld_num = p.hhld_num AND main.pers_num = p.pers_num
	AND (main.n_pers_trips IS NULL OR main.n_tran_trips IS NULL OR
        main.sex IS NULL OR main.driver_lic IS NULL OR
        main.pass_ttc IS NULL OR main.pass_go IS NULL OR
        main.pass_oth IS NULL OR main.emp_ft IS NULL OR
        main.emp_workhome IS NULL OR main.student IS NULL OR
        main.free_park IS NULL OR main.occupation IS NULL OR
        main.emp_region IS NULL OR main.sch_region IS NULL OR
        main.emp_pd IS NULL OR main.sch_pd IS NULL);

UPDATE datatable main
SET hhld_pers = CASE WHEN h.n_person < 99 THEN h.n_person
        ELSE 0 END,
	hhld_veh = CASE WHEN h.n_vehicle < 99 THEN h.n_vehicle
        ELSE 0 END,
    hhld_lic = CASE WHEN h.n_licence < 99 THEN h.n_licence
        ELSE 0 END,
    hhld_emp_ft = CASE WHEN h.n_emp_ft < 99 THEN h.n_emp_ft
        ELSE 0 END,
    hhld_emp_pt = CASE WHEN h.n_emp_pt < 99 THEN h.n_emp_pt
        ELSE 0 END,
    hhld_stu = CASE WHEN h.n_student < 99 THEN h.n_student
        ELSE 0 END,
    hhld_trips = CASE WHEN h.n_hhld_tri < 99 THEN h.n_hhld_tri
        ELSE 0 END,
    hhld_type = CASE WHEN h.dwell_type = 1 THEN 1
        WHEN h.dwell_type = 2 THEN -1
    	ELSE 0 END,
    hhld_region = CASE WHEN h.region_hhl <= 6 THEN h.region_hhl
        ELSE 0 END,
    hhld_pd = CASE WHEN h.pd_hhld = 1 THEN 1
        WHEN h.pd_hhld = 2 OR h.pd_hhld = 4 OR h.pd_hhld = 6 THEN 2
        WHEN h.pd_hhld = 3 THEN 3
        WHEN h.pd_hhld = 5 THEN 4
        WHEN h.pd_hhld = 10 OR h.pd_hhld = 11 OR h.pd_hhld = 12 THEN 5
        WHEN h.pd_hhld = 7 OR h.pd_hhld = 8 OR h.pd_hhld = 9 THEN 6
        WHEN h.pd_hhld > 12 AND h.pd_hhld <= 16 THEN 7
        WHEN h.pd_hhld > 16 AND h.pd_hhld <= 46 THEN 8
        ELSE 0 END
FROM (SELECT * FROM tts11.hhld) h
WHERE main.hhld_num = h.hhld_num
    AND (main.hhld_pers IS NULL OR main.hhld_veh IS NULL OR
        main.hhld_lic IS NULL OR main.hhld_emp_ft IS NULL OR
        main.hhld_emp_pt IS NULL OR main.hhld_stu IS NULL OR
        main.hhld_trips IS NULL OR main.hhld_type IS NULL OR
        main.hhld_region IS NULL OR main.hhld_pd IS NULL);

UPDATE datatable main
SET n_go_rail = CASE WHEN r.n_go_rail < 99 THEN r.n_go_rail
        ELSE 0 END,
	n_go_bus = CASE WHEN r.n_go_bus < 99 THEN r.n_go_bus
        ELSE 0 END,
    n_ttc_sub = CASE WHEN r.n_subway < 99 THEN r.n_subway
        ELSE 0 END,
	n_ttc_bus = CASE WHEN r.n_ttc_bus < 99 THEN r.n_ttc_bus
        ELSE 0 END,
    n_local = CASE WHEN r.n_local < 99 THEN r.n_local
        ELSE 0 END,
    n_other = CASE WHEN r.n_other < 99 THEN r.n_other
        ELSE 0 END,
    use_ttc = CASE WHEN r.use_ttc = 'Y' THEN 1
        WHEN r.use_ttc = 'N' THEN -1
        ELSE 0 END,
    trans_accs_m = CASE WHEN r.mode_accs = 'W' THEN 1
        WHEN r.mode_accs = '9' THEN 0
        ELSE -1 END,
    trans_egrs_m = CASE WHEN r.mode_egrs = 'W' THEN 1
        WHEN r.mode_egrs = '9' THEN 0
        ELSE -1 END,
    trans_accs_reg = CASE WHEN r.region_acc <= 6 THEN r.region_acc
        ELSE 0 END,
    trans_egrs_reg = CASE WHEN r.region_egr <= 6 THEN r.region_egr
        ELSE 0 END,
    trans_accs_pd = CASE WHEN r.pd_accs = 1 THEN 1
        WHEN r.pd_accs = 2 OR r.pd_accs = 4 OR r.pd_accs = 6 THEN 2
        WHEN r.pd_accs = 3 THEN 3
        WHEN r.pd_accs = 5 THEN 4
        WHEN r.pd_accs = 10 OR r.pd_accs = 11 OR r.pd_accs = 12 THEN 5
        WHEN r.pd_accs = 7 OR r.pd_accs = 8 OR r.pd_accs = 9 THEN 6
        WHEN r.pd_accs > 12 AND r.pd_accs <= 16 THEN 7
        WHEN r.pd_accs > 16 AND r.pd_accs <= 46 THEN 8
        ELSE 0 END,
    trans_egrs_pd = CASE WHEN r.pd_egrs = 1 THEN 1
        WHEN r.pd_egrs = 2 OR r.pd_egrs = 4 OR r.pd_egrs = 6 THEN 2
        WHEN r.pd_egrs = 3 THEN 3
        WHEN r.pd_egrs = 5 THEN 4
        WHEN r.pd_egrs = 10 OR r.pd_egrs = 11 OR r.pd_egrs = 12 THEN 5
        WHEN r.pd_egrs = 7 OR r.pd_egrs = 8 OR r.pd_egrs = 9 THEN 6
        WHEN r.pd_egrs > 12 AND r.pd_egrs <= 16 THEN 7
        WHEN r.pd_egrs > 16 AND r.pd_egrs <= 46 THEN 8
        ELSE 0 END
FROM (SELECT * FROM tts11.tran) r
WHERE main.hhld_num = r.hhld_num AND
    main.pers_num = r.pers_num AND
    main.trip_num = r.trip_num
    AND (main.n_go_rail IS NULL OR main.n_go_bus IS NULL OR
        main.n_ttc_sub IS NULL OR main.n_ttc_bus IS NULL OR
        main.n_local IS NULL OR main.n_other IS NULL OR
        main.use_ttc IS NULL OR main.trans_accs_m IS NULL OR
        main.trans_egrs_m IS NULL OR main.trans_accs_reg IS NULL OR
        main.trans_egrs_reg IS NULL OR main.trans_accs_pd IS NULL OR
        main.trans_egrs_pd IS NULL);

SELECT * FROM datatable ORDER BY gid;
