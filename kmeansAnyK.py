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

colors = [ "Red", "Blue", "Cyan", "Yellow", "Green", "Magenta", "Black"]

def plotPoint(xcol, ycol, clusterNum):
	
	plt.scatter(xcol, ycol, 13, c=colors[clusterNum], marker ="x")
		
	return plt

def plotPoints(stanData, xcol, ycol, clusterNum):
	
	for row in stanData:
		x = row[xcol]
		y = row[ycol]		
		plt.scatter(x, y, 13, c=colors[clusterNum], marker ="x")
		
	return plt
	
def plotCluster(cluster, clusterNum):

	x = cluster[1]
	y = cluster[2]
	plt.scatter(x, y, 35, c=colors[clusterNum])
		
	return plt 


def purity(clusters):
		
	avgPurity = 0
	N = 0
	for cluster in clusters:
		clusterSize = len(clusters[cluster])
		positive = 0
		negative = 0
		for point in clusters[cluster]:
			N += 1
			n = int(point[0])
			if (n == 1):
				positive += 1
			else:
				negative += 1
			
		clusterPurity = float(1)/clusterSize * max(positive, negative)
		avgPurity += float(clusterSize)*clusterPurity
	
	return avgPurity/N
	
def sumOfMagChange(clustersMean, prevClustMeans):

	sum = 0
	for clusterMean in clustersMean.keys():
		x = clustersMean[clusterMean][0]
		y = clustersMean[clusterMean][1]
		x_1 = prevClustMeans[clusterMean][0]
		y_1 = prevClustMeans[clusterMean][1]
		d1 = x-x_1 if x-x_1 > 0 else x_1-x
		d2 = y-y_1 if y-y_1 > 0 else y_1-y
		sum += d1 + d2

	return sum
	
def calculateClusterMeans(clusters):

	newClusterMeans = {}
	for clusterNum in clusters.keys():
		x_sum = 0
		y_sum = 0
		for point in clusters[clusterNum]:
			x_sum += point[1]
			y_sum += point[2]
		
		mean = [x_sum/len(clusters[clusterNum]), y_sum/len(clusters[clusterNum])]
		newClusterMeans[clusterNum] = mean
	
	return newClusterMeans
	
	
def kmeans(stanData, k, xcol, ycol):

	#
	#	plot the initial plot and initial clusters
	#
	plotPoints(stanData, xcol, ycol, 0)

	clusters = {}
	clustersMean = {}
	clusterNum = 1
	for i in range(k):
		rand = random.randint(1, len(stanData)-1)
		x = stanData[rand][xcol]
		y = stanData[rand][ycol]
		point = [stanData[rand][0], x, y]
		plotCluster(point, 1)
		temp = []
		temp.append(point)
		clusters[clusterNum] = temp
		clustersMean[clusterNum] = [x, y]
		clusterNum += 1
		
	plt.savefig("intial.png") 
	plt.clf()
	
	#
	#	assign and plot data to initial clusters
	#
	for row in stanData:
		x = row[xcol]
		y = row[ycol]
		temp = dict()
		
		for clusterNum in clustersMean.keys():
			x1 = clustersMean[clusterNum][0]
			y1 = clustersMean[clusterNum][1]
			d = math.sqrt((x-x1)**2 + (y-y1)**2)
			temp[clusterNum] = d			
		temp = sorted(temp.iteritems(), key=lambda (k,v): (v,k))	
		
		classRow = row[0]
		tempList = [classRow, x, y]
		clusterAssigned = temp[0][0]
		clusters[clusterAssigned].append(tempList)

		plotPoint(x, y, clusterAssigned-1)
	
	#plot the Clusters
	for clusterNum in clustersMean.keys():
		x = clustersMean[clusterNum][0]
		y = clustersMean[clusterNum][1]
		plotCluster(['NaN', x, y], clusterNum-1)
			
	plt.savefig("intial_clusters.png") 
	plt.clf()
	
	#
	# update the new cluster means
	#
	prevClustMeans = clustersMean
	clustersMean = calculateClusterMeans(clusters)
	
	#
	#	plot the final cluster assignments
	#
	e = 2**(-23)
	iter = 0
	while sumOfMagChange(clustersMean, prevClustMeans) > e:
	
		clusters = {}
		for clusterNum in clustersMean.keys():
			clusters[clusterNum] = []
			
		for row in stanData:
			x = row[xcol]
			y = row[ycol]
			
			temp = dict()
			
			for clusterNum in clustersMean.keys():
				x1 = clustersMean[clusterNum][0]
				y1 = clustersMean[clusterNum][1]
				d = math.sqrt((x-x1)**2 + (y-y1)**2)
				temp[clusterNum] = d			
			temp = sorted(temp.iteritems(), key=lambda (k,v): (v,k))	
			
			classRow = row[0]
			tempList = [classRow, x, y]
			clusterAssigned = temp[0][0]
			clusters[clusterAssigned].append(tempList)

		prevClustMeans = clustersMean
		clustersMean = calculateClusterMeans(clusters)
		iter +=1
		
	#plot the points
	for clusterNum in clusters.keys():
		for i in range(len(clusters[clusterNum])):
			x = clusters[clusterNum][i][1]
			y = clusters[clusterNum][i][2]
			plotPoint(x, y, clusterNum-1)
		
	#plot the Clusters
	for clusterNum in clustersMean.keys():
		x = clustersMean[clusterNum][0]
		y = clustersMean[clusterNum][1]
		plotCluster(["NaN", x, y], clusterNum-1)
			
	plt.savefig("final_clusters.png") 
	plt.clf()	
	
	print "Number of iterations: ", iter	
	print "Purity: ", purity(clusters)

def standardizeData(data, specs):

	#
	#	assign k and graph columns
	#
	k = int(specs[0])
	xcol = int(specs[1])
	ycol = int(specs[2])
		
	#
	#	calculate mean and standard deviation for each column
	#
	mean = [0]
	std = [0]
	for i in range(1, len(data[0])):
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
		tempCol = [row[0]]
		for i in range(1, len(row)):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		stanData.append(tempCol)

	kmeans(stanData, k, xcol, ycol)
	
def parseData(specs):

	filename = "diabetes.csv"
	with open(filename, 'rb') as f:
		reader = csv.reader(f)		
		#
		# 	read in the data
		#
		data = []
		for row in reader:
			data.append(row);
			
	standardizeData(data, specs)

def main():

	if len(sys.argv) != 4:
		print 'Please choose k, feauture 1 and feauture 2 (in that order)'
	
	else:
		k = sys.argv[1];
		feauture1 = sys.argv[2]
		feauture2 = sys.argv[3]
		specs = [k, feauture1, feauture2]
		parseData(specs)

main()