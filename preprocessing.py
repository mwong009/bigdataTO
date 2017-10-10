import numpy as np
import pandas as pd
from six.moves import cPickle

def extractdata(csvName='datatable.csv', type='MNL'):
	""" string csvName: Name of csv file. e.g. 'datatable.csv'
	"""
	df = pd.read_csv(csvName)

	y = df.mode_prime.values

	# scale data
	x1 = df[['age', 'n_pers_trips', 'n_tran_trips', 'hhld_pers',
       'hhld_veh', 'hhld_lic', 'hhld_emp_ft', 'hhld_emp_pt', 'hhld_stu',
       'hhld_trips', 'n_go_rail', 'n_go_bus', 'n_ttc_bus', 'n_ttc_sub',
       'n_local', 'n_other', 'trip_km', 'car_pool']].values
	# normalize
	x1 = x1/x1.max(axis=0)

	# binary data
	x2 = df[['sex', 'driver_lic', 'pass_ttc', 'pass_go', 'pass_oth',
		'emp_ft', 'emp_workhome', 'student', 'free_park',
		'hhld_type', 'use_ttc', 'hwy407', 'trans_accs_m',
		'trans_egrs_m', 'trip_type']].values

	# categorical data 1
	x31 = df[['occupation', 'trip_purp']].values
	# one-hot
	x31b = np.eye(x31.max()+1)[x31][:,1:]

	# categorical data 2
	x32 = df[['emp_region', 'sch_region', 'hhld_region', 'trans_accs_reg',
		'trans_egrs_reg', 'trip_orig_reg', 'trip_dest_reg']].values
	# one-hot
	x32b = np.eye(x32.max()+1)[x32][:,:,1:]

	# categorical data 3
	x33 = df[['emp_pd', 'sch_pd', 'hhld_pd', 'trans_accs_pd', 'trans_egrs_pd',
		'trip_orig_pd', 'trip_dest_pd']].values
	# one-hot
	x33b = np.eye(x33.max()+1)[x33][:,:,1:]

	if type == 'MNL':
		dataset = {'y': y, 'scale_data': x1, 'binary_data': x2,
			'occupation': x31b, 'region': x32b, 'pd': x33b}
	else:
		dataset = {'y': y, 'scale_data': x1, 'binary_data': x2,
			'occupation': x31, 'region': x32, 'pd': x33}

	with open('dataset.save', 'wb') as f:
		cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	extractdata()
