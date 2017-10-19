# in processing.py
scale_norm = {'scale_variables': scale_max, 'trip_km': trip_km_max}

# in script_rbm.py
rbm.load_variables(..., validate_terms = {'mode_prime': 'category', ..., 'trip_km': 'scale'})

# in RBM.py
self.norms = norms
self.validate_terms = validate_terms

# in sample_v_given_h

if ndim == 2:
  # two types: scale and binary
  if name in self.validate_terms:
    
