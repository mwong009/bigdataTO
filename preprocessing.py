import numpy as np
import pandas as pd
import pickle
import sys
import collections

def main(argv):
	csv_name = argv[1]

	df = pd.read_csv(csv_name)
	num_rows = df.count()[0]

	scale_variables = df[['age', 'n_pers_trips', 'n_tran_trips', 'hhld_pers',
    	'hhld_veh', 'hhld_lic', 'hhld_emp_ft', 'hhld_emp_pt', 'hhld_stu',
    	'hhld_trips', 'n_go_rail', 'n_go_bus', 'n_ttc_bus', 'n_ttc_sub',
    	'n_local', 'n_other', 'car_pool']].values

	scale_norms = scale_variables.max(axis=0)
	scale_variables = scale_variables/scale_norms

	binary_variables = df[['sex', 'driver_lic', 'pass_ttc', 'pass_go',
		'pass_oth', 'emp_ft', 'emp_workhome', 'student', 'free_park',
		'hhld_type', 'use_ttc', 'hwy407', 'trans_accs_m', 'trans_egrs_m',
		'trip_type']].values

	occupation_variables = df[['occupation']].values

	occupation_variables = np.eye(occupation_variables.max()+1)[
		occupation_variables]

	region_variables = df[['emp_region', 'sch_region', 'hhld_region',
		'trans_accs_reg', 'trans_egrs_reg', 'trip_orig_reg',
		'trip_dest_reg']].values

	region_variables = np.eye(region_variables.max()+1)[region_variables]

	pd_variables = df[['emp_pd', 'sch_pd', 'hhld_pd', 'trans_accs_pd',
		'trans_egrs_pd', 'trip_orig_pd', 'trip_dest_pd']].values

	pd_variables = np.eye(pd_variables.max()+1)[pd_variables]

	trip_km = df[['trip_km']].values

	trip_km_norms = trip_km.max(axis=0)
	trip_km = trip_km/trip_km_norms

	trip_purp = df[['trip_purp']].values
	trip_purp = np.eye(trip_purp.max()+1)[trip_purp]

	mode_prime = df[['mode_prime']].values
	mode_prime = np.eye(mode_prime.max()+1)[mode_prime]

	dataset = [('mode_prime', mode_prime),
		('trip_purp', trip_purp),
		('trip_km', trip_km),
		('scale_variables', scale_variables),
		('binary_variables', binary_variables),
		('occupation_variables', occupation_variables),
		('region_variables', region_variables),
		('pd_variables', pd_variables)]

	dataset = collections.OrderedDict(dataset)

	norms = {'scale_norms': scale_norms, 'trip_km_norms': trip_km_norms}

	for name, item in dataset.items():
		print(name, item.shape)
	for name, item in norms.items():
		print(name, item)

	with open('dataset.save', 'wb') as f:
		pickle.dump((dataset, norms), f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		raise SyntaxError("Not enough arguments: *.csv")
	elif len(sys.argv) == 2:
		main(sys.argv)
	else:
		raise SyntaxError("Too many arguments")
