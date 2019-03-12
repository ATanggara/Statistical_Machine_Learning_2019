#
# Statistical machine learning lab exam.
#
# Instructions:
#
#  - do not edit this file before "BEGIN_SOLUTION" or after "END_SOLUTION"
#  - do not import any modules or external code
#
# You will get zero marks for this part if you break these rules.
#

import numpy as np


def linear_least_squares(x, y):

	################################################
	#
	# INPUTS:
	#	x: input data, shape d by n
	#   y: output targets, shape n by 1
	#
	#  OUTPUTS:
	#   w: linear least squares weights, d by 1
	#
	################################################

	# BEGIN_SOLUTION 
    x = x.T
    w = np.linalg.inv(x.T@x) @ x.T @ y
    print(w)
	# END_SOLUTION

    return w
