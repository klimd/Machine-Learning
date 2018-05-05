#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')

import sys
import os
import matplotlib.pyplot as plt
import csv
import numpy


def standardizeData(data, colNum, rowNum):

	#
	#	calculate mean and standard deviation for each column
	#
	mean = [0]
	std = [0]
	for x in range (1,colNum):
		mean.append(numpy.mean(data[x]))
		std.append(numpy.std(data[x]))
	
	#
	#	standardize the data
	#
	count = 1
	for column in data[1:]:
		for i in range (0, len(column)):
			column[i] = (float(column[i]) - mean[count])/std[count]
		count+=1
	
	#
	#	store it back into a list of rows
	#
	stanData = []
	covMatrixData = []
	
	for row in range(rowNum):
		l = []
		sl = []
		for column in range(colNum):
			l.append(data[column][row])
			if column != 0 :
				sl.append(data[column][row])
			
		stanData.append(l)
		covMatrixData.append(sl)
	
	#
	# compute covariance matrix
	#
	covMatrix = numpy.cov(covMatrixData, rowvar=False)
	
	#
	# compute Eigen vectors and Eigen values
	#	and select the 2 max 
	
	eValue, eVector = numpy.linalg.eig(covMatrix);	
	 
	max1 = max(eValue);
	max2 = -float("inf")
	index = 0
	for x in eValue:
		if x > max2 and x != max1:
			max2 = x
			max2_index = index
			
		if x == max1:
			max1_index = index
		
		index += 1
		
	max1Vector = eVector[max1_index]
	max2Vector = eVector[max2_index]
	
	#
	# compute data for plotting
	#
	
	plotData = []
	index = 0
	for x in covMatrixData:
		
		matrixA = numpy.dot(x, numpy.transpose(max1Vector))
		matrixB = numpy.dot(x, numpy.transpose(max2Vector))
		
		points = [matrixA, matrixB]
		pair = [stanData[index][0] , points]
		plotData.append(pair)
		index += 1
	
	for x in plotData: 
		if x[0] == 1:
			plt.scatter(x[1][0],x[1][1], c="Red", facecolors='none', edgecolors='r')
		else:
			plt.scatter(x[1][0],x[1][1], c="Blue", marker ="x", facecolors='none', edgecolors='r')

	
	plt.axis([-3, 4, -6, 6])
	plt.savefig("p1.png") 
	
	
def parseData(filename):

	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		colNum = len(next(reader))
		f.seek(0)
		
		#
		#	create data = [[...] [...] ... [...]]
		#
		data = []
		for x in range (0,colNum):
			data.append([])
				
		#
		# 	separate by columns
		#
		rowNum = 0
		for row in reader:
			for i in range(0,colNum):
				data[i].append(float(row[i]))
			rowNum += 1
			
	standardizeData(data, colNum, rowNum)

def main():

	if len(sys.argv) != 2:
		print 'No filename specified'
	
	else:
		filename = sys.argv[1];
		parseData(filename)

main()