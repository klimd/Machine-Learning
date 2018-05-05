#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')

import sys
import os
import matplotlib.pyplot as plt
import csv
import numpy
import random
import math

def standardizeData(data, colNum, rowNum):

	#
	#	calculate mean and standard deviation for each column
	#
	mean = [0]
	std = [0]
	for i in range(1, colNum):
		tempCol = [0]
		for row in data:
			tempCol.append(float(row[i]))
			
		mean.append(numpy.mean(tempCol))
		std.append(numpy.std(tempCol)) 

	#
	#	standardize the data
	#
	stanData = []
	for row in data:
		tempCol = [0]
		for i in range(1, len(row)):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		stanData.append(tempCol)

	#
	#	plot the initial step
	#
	for row in stanData:
		x = row[7]
		y = row[6]		
		plt.scatter(x, y, 13, c="Red", marker ="x")
	
	#
	#	plot the initial clusters
	#
	k = 2
	clusters = []
	for i in range(k):
		rand = random.randint(0, rowNum)
		cluster = [stanData[rand][7], stanData[rand][6]]
		clusters.append(cluster)
		plt.scatter(cluster[0], cluster[1], 35, c="Blue")
		
	plt.axis([-2, 6, -5, 5])
	plt.savefig("intial.png") 
	
	#
	#	plot the initial cluster assignments
	#
	cluster1 = []
	cluster2 = []
	plt.clf()
	for row in stanData:
		x = row[7]
		y = row[6]
		
		dist = []
		for cluster in clusters:
			x1 = cluster[0]
			y1 = cluster[1]
			d = math.sqrt( (x-x1)**2 + (y-y1)**2)
			dist.append(d)
		
		temp = [x, y]
		if dist[0] < dist[1]:
			cluster1.append(temp)
			plt.scatter(x, y, 13, c="Red", marker ="x")
		else:
			cluster2.append(temp)
			plt.scatter(x, y, 13, c="Blue", marker ="x")		
		
	plt.scatter(clusters[0][0], clusters[0][1], 35, c="Red")
	plt.scatter(clusters[1][0], clusters[1][1], 35, c="Blue")
		
	plt.axis([-2, 6, -5, 5])
	plt.savefig("initial_cluster.png") 
	
	#
	#	plot the final cluster assignments
	#
	plt.clf()
	e = 2**(-23)
	iter = 0
	cMeans = clusters
	stop = False
	while stop != True:
		iter += 1
		prevMeans = cMeans
		cluster1 = []
		cluster2 = []
		for row in stanData:
			x = row[7]
			y = row[6]
			
			dist = []
			for cmean in cMeans:
				x1 = cmean[0]
				y1 = cmean[1]
				d = math.sqrt( (x-x1)**2 + (y-y1)**2)
				dist.append(d)
			
			temp = [x, y]
			if dist[0] < dist[1]:
				cluster1.append(temp)
			else:
				cluster2.append(temp)
		
		cMeans = []
		mean = [0, 0]
		for point in cluster1:
			mean[0] += point[0]
			mean[1] += point[1]
		cMean1 = [mean[0]/len(cluster1), mean[1]/len(cluster1)]
		cMeans.append(cMean1)
	
		mean = [0, 0]
		for point in cluster2:
			mean[0] += point[0]
			mean[1] += point[1]
		cMean2 = [mean[0]/len(cluster2), mean[1]/len(cluster2)]
		cMeans.append(cMean2)
		
		sum = 0
		for i in range(len(cMeans)):
			x_1 = cMeans[i][0]
			y_1 = cMeans[i][1]
			x = prevMeans[i][0]
			y = prevMeans[i][1]
			d1 = x-x_1 if x-x_1 > 0 else x_1-x
			d2 = y-y_1 if y-y_1 > 0 else y_1-y
			sum+= d1+d2
		
		if (sum < e):
			stop = True
	
	plt.clf()		
	for point in cluster1:
		plt.scatter(point[0], point[1], 13, c="Red", marker ="x")

	for point in cluster2:
		plt.scatter(point[0], point[1], 13, c="Blue", marker ="x")
		
	plt.scatter(prevMeans[0][0], prevMeans[0][1], 35, c="Red")
	plt.scatter(prevMeans[1][0], prevMeans[1][1], 35, c="Blue")
	
	plt.axis([-2, 6, -5, 5])
	plt.savefig("final_cluster.png") 
	
	#
	#	calculate the purity
	#
	
	print iter
	
def parseData(filename):

	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		colNum = len(next(reader))
		f.seek(0)
		
		#
		# 	read in the data
		#
		data = []
		rowNum = 0
		for row in reader:
			data.append(row);
			rowNum +=1
			
	standardizeData(data, colNum, rowNum)

def main():

	if len(sys.argv) != 2:
		print 'No filename specified'
	
	else:
		filename = sys.argv[1];
		parseData(filename)

main()