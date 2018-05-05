#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')

import sys
import os
import matplotlib.pyplot as plt

import csv
import numpy

def main():

	Xt = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[-0.2602, -0.9697, -0.4967, 0.2129, -1.6792, -0.2602, 0.4493, 1.3954, -0.0236, 1.6319]]
	X = [[1, -0.2602], [1, -0.9697], [1, -0.4967], [1, 0.2129], [1, -1.6792], [1, -0.2602], [1, 0.4493], [1, 1.3954], [1, -0.0236], [1, 1.6319]]
	Y = [[-2], [-5], [-3], [0], [-8], [-2], [1], [5], [-1], [6]]
	
	Xt = numpy.asmatrix(Xt)
	X = numpy.asmatrix(X)
	Y = numpy.asmatrix(Y)
	#print  X
	print 
	#print Xt
	print
	#print numpy.transpose(X)
#	print numpy.dot(Xt, X)
	#print numpy.linalg.inv(numpy.dot(Xt, X))
	#print numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(Xt, X)), Xt), Y)
	
	inverse = [[10, 0], [0, 9.01423]]
	inverse = numpy.asmatrix(inverse)
	
#	print numpy.linalg.inv(inverse)
	
	print numpy.dot(numpy.linalg.inv(inverse), Xt)
	
	print numpy.dot(numpy.dot(numpy.linalg.inv(inverse), Xt), Y)
main()