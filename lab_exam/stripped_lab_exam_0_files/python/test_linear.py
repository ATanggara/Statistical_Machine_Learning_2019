import numpy as np
import traceback
import sys

import linear


def test_linear(x, y, w_true):

	################################################
	#
	# INPUTS:
	#	x: input data, shape d by n
	#   y: output targets, shape n by 1
	#   w_true: true weights, shape d by 1, or None
	#
	#  OUTPUTS:
	#   w: linear regression weights, d by 1
	#
	################################################

	d, n = x.shape
	assert y.shape == (n, 1)

	try:
		# call the student's code:
		w_student = linear.linear_least_squares(x, y)
	except:
		# the student's code raised an exception:
		traceback.print_exc()
		return False


	assert w_student.shape == (d, 1)


	if w_true is not None:
		# check if the correct weights have been calculated:
		w_matches = np.allclose(w, w_student)
		print('w_matches', w_matches)
		return w_matches
	else:
		# check if the predictions match on the training data:
		y_student = np.dot(x.T, w_student)
		y_matches = np.allclose(y, y_student)
		print('y_matches', y_matches)
		return y_matches


def random_data(d, n):
	print('d = %i n = %i' % (d, n))
	# linear model weights:
	w_true = np.random.randn(d, 1)
	# input datapoints:
	x = np.random.randn(d, n)
	# scalar outputs:
	y = np.dot(x.T, w_true)
	return x, y, w_true


if __name__ == '__main__':

	total_marks = 0

	print('First test:')
	x, y, w = random_data(d=2, n=3)
	if test_linear(x=x, y=y, w_true=w):
		print('passed')
		total_marks = total_marks + 1
	else:
		print('failed')
	print()

	print('Second test:')
	x, y, w = random_data(d=2, n=1)
	if test_linear(x=x, y=y, w_true=None):
		print('passed')
		total_marks = total_marks + 1
	else:
		print('failed')
	print()

	print('Tests completed, total_marks =', total_marks)


