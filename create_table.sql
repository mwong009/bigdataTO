CREATE TABLE datatable
(
	gid integer,
   	trip_num integer,
    hhld_num integer,
    pers_num integer,
    
    mode_prime integer, -- 0: Unknown, 1: Public Transit(B), 2: Bicycle(C), 3: AutoDriver(D), 4: Go Rail(G), 5: Joint Go/Transint(J)
    					-- 6: Motocycle(M), 7: Other(O), 8: Auto Passenger(P), 9: School Bus(S), 10: Taxi(T), 11: Walk(W) (SOFTMAX)
    
    age integer, -- 1-98: Age, 0: Unknown
    n_pers_trips integer, -- 1-98: Number of Trips made on day
    n_tran_trips integer, -- 1-98: Number of Transit Trips made on day
    hhld_pers integer, -- 1-98: Number of persons in dwelling
    hhld_veh integer, -- 0-98: Number of vehicles in dwelling
    hhld_lic integer, -- 0-98 Number of driver licences in dwelling
    hhld_emp_ft integer, -- 0-98: Number of Full Time workers in dwelling
    hhld_emp_pt integer, -- 0-98: Number of Part Time workers in dwelling
    hhld_stu integer, -- 0-98: Number of Students in dwelling
    hhld_trips integer, -- 0-98: Number of Dwelling trips made on day
    n_go_rail integer, -- 1-98: Number of GO Rail Routes used
    n_go_bus integer, -- 1-98: Number of GO Bus Routes used
    n_ttc_bus integer, -- 1-98: Number of TTC Bus Routes used
    n_ttc_sub integer, -- 1-98: Number of TTC Subway Routes used
    n_local integer, -- 1-98: Number of Local Transit (non-TTC, non-GO) Routes used
    n_other integer, -- 1-98: Number of Other Transit Routes used
    trip_km integer, -- 0-998: Straight Line trip length (meters)
    car_pool integer, -- 1-98: Number of people in vehicle
    
    sex integer, -- 1: Female, -1: Male, 0: Unknown   
    driver_lic integer, -- 1: Yes, -1: No, 0: Unknown
    pass_ttc integer, -- 1: TTC Pass, -1: None, 0: Unknown
    pass_go integer, -- 1: Go Transit Pass, -1: None, 0: Unknown
    pass_oth integer, -- 1: Other Agency Pass, -1: None, 0: Unknown
    emp_ft integer, -- 1: Full Time, -1: Part Time, 0: Unknown
    emp_workhome integer, -- 1: Work at Home, -1: Not Work at Home, 0: Unknown
    student integer, -- 1: Full Time Student, -1: Part Time Student, 0: Unknown/Not a Student
    free_park integer, -- 1: Yes, -1: No, 0: Unknown
    hhld_type integer, -- 1: House, -1: Apartment, 0: Other/Unknown
    use_ttc integer, -- 1: Yes, -1: No, 0: N/A, Unknown
    hwy407 integer, -- 1: Used Hwy 407, -1: No, 0: N/A,Unknown
    trans_accs_m integer, -- 1: Walk(W), -1: Other, 0: Unknown
    trans_egrs_m integer, -- 1: Walk(W), -1: Other, 0: Unknown
    trip_type integer, -- 1: Home-Work or Home-School, -1: Non-home-based, 0: Other/Unknown
    
    occupation integer, -- 0: Unknown, 1: General Office, 2: Manufacturing, 3: Professional, 5: Retail 6: Unemployed (SOFTMAX)
       
    emp_region integer, -- 0: Unknown,1: City of Toronto, 2: Durham, 3: York, 4: Peel, 5: Halton, 6: Hamilton (SOFTMAX)
   	sch_region integer, -- 0: Unknown, 1: City of Toronto, 2: Durham, 3: York, 4: Peel, 5: Halton, 6: Hamilton (SOFTMAX)
    hhld_region integer, -- 0: Unknown, 1: City of Toronto, 2: Durham, 3: York, 4: Peel, 5: Halton, 6: Hamilton (SOFTMAX)
    trans_accs_reg integer, -- 0: Unknown, 1: City of Toronto, 2: Durham, 3: York, 4: Peel, 5: Halton, 6: Hamilton (SOFTMAX)
    trans_egrs_reg integer, -- 0: Unknown, 1: City of Toronto, 2: Durham, 3: York, 4: Peel, 5: Halton, 6: Hamilton (SOFTMAX)
    trip_orig_reg integer, -- 0: Unknown, 1: City of Toronto, 2: Durham, 3: York, 4: Peel, 5: Halton, 6: Hamilton (SOFTMAX)
    trip_dest_reg integer, -- 0: Unknown, 1: City of Toronto, 2: Durham, 3: York, 4: Peel, 5: Halton, 6: Hamilton (SOFTMAX)
    
    emp_pd integer, -- 0: Unknown, 1: Toronto Downtown Core (PD 1), 2: Rest of South Toronto (PD 2,4,6), 3: York (PD 3), 
    				-- 4: East York (PD 5), 5: North York (PD 10-12), 6: Etobicoke (PD 7-9), 7: Scarborough (PD 13-16), 8: Rest of GTHA (PD 17-46) (SOFTMAX)
    sch_pd integer, -- 0: Unknown, 1: Toronto Downtown Core (PD 1), 2: Rest of South Toronto (PD 2,4,6), 3: York (PD 3), 
    				-- 4: East York (PD 5), 5: North York (PD 10-12), 6: Etobicoke (PD 7-9), 7: Scarborough (PD 13-16), 8: Rest of GTHA (PD 17-46) (SOFTMAX)
    hhld_pd integer, -- 0: Unknown, 1: Toronto Downtown Core (PD 1), 2: Rest of South Toronto (PD 2,4,6), 3: York (PD 3), 
    				-- 4: East York (PD 5), 5: North York (PD 10-12), 6: Etobicoke (PD 7-9), 7: Scarborough (PD 13-16), 8: Rest of GTHA (PD 17-46) (SOFTMAX)
    trans_accs_pd integer, -- 0: Unknown, 1: Toronto Downtown Core (PD 1), 2: Rest of South Toronto (PD 2,4,6), 3: York (PD 3), 
    					   -- 4: East York (PD 5), 5: North York (PD 10-12), 6: Etobicoke (PD 7-9), 7: Scarborough (PD 13-16), 8: Rest of GTHA (PD 17-46) (SOFTMAX)
    trans_egrs_pd integer, -- 0: Unknown, 1: Toronto Downtown Core (PD 1), 2: Rest of South Toronto (PD 2,4,6), 3: York (PD 3), 
    					   -- 4: East York (PD 5), 5: North York (PD 10-12), 6: Etobicoke (PD 7-9), 7: Scarborough (PD 13-16), 8: Rest of GTHA (PD 17-46) (SOFTMAX)
    trip_orig_pd integer, -- 0: Unknown, 1: Toronto Downtown Core (PD 1), 2: Rest of South Toronto (PD 2,4,6), 3: York (PD 3), 
    				      -- 4: East York (PD 5), 5: North York (PD 10-12), 6: Etobicoke (PD 7-9), 7: Scarborough (PD 13-16), 8: Rest of GTHA (PD 17-46) (SOFTMAX)    
    trip_dest_pd integer, -- 0: Unknown, 1: Toronto Downtown Core (PD 1), 2: Rest of South Toronto (PD 2,4,6), 3: York (PD 3), 
    				      -- 4: East York (PD 5), 5: North York (PD 10-12), 6: Etobicoke (PD 7-9), 7: Scarborough (PD 13-16), 8: Rest of GTHA (PD 17-46) (SOFTMAX)   
)
