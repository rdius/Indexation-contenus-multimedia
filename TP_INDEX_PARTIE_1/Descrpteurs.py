###UTILISATION##################################
#
# python Descripteurs.py 'nom_image_requette'
###############################################""


## importation des packages
import pickle
from scipy.spatial import distance as dist
import numpy as np
np.seterr(over='ignore')
import argparse
import glob
import cv2
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import os
from math import sqrt
import csv
import random
import math
import operator

	#fonction de calcul d'histogramme 
def histogramme(rep, img):
	
	nb_bits=8
	
	x=1
	hist=[]
	#lecture des images de la base  
	img = cv2.imread(rep + img)
	#split de la base de couleur RGB
	b, g, r = cv2.split(img)
	# histogramme de B 
	histrb = cv2.calcHist([img],[0],None,[nb_bits],[0,nb_bits])
	#histogramme de G 
	histrg = cv2.calcHist([img],[1],None,[nb_bits],[0,nb_bits])
	#histogramme R
	histrr = cv2.calcHist([img],[2],None,[nb_bits],[0,nb_bits])
	#sommation des histogramme de RGB 
	for i in range(len(histrb)):
		hist.append( float(histrb[i]))
	for i in range(len(histrg)):
		hist.append( float(histrg[i]))
	for i in range(len(histrr)):
		hist.append( float(histrr[i]))
	return hist # returne la valeur de l'histogramme des trois couleurs 
# Calcul de la distancee entre deux images de la base 
def distance(histA, histB):
	histg=0
	d_global=[] # liste des distances 
	#parcour des valeurs des histogrammes A t B
	for i in range(len(histA)):

		histg = abs(float(histA[i])- float(histB[i]))
		d_global.append( histg/sum(histA))# insere toutes las valeurs dans la liste 
		
	return	 float(sum(d_global)) # retourne la distance entre les deux histogrammes

##Implementation manuelle du moment de Hu "reference wikippedia"
"""
def BasicMoment(p,q,img):
	mpq=0
	
	b=[]
	x, y, c= img.shape
	for i in range(x):
		for j in range(y):
			mpq +=(img[i,j]*(i**p)*(j**q))					
	return mpq   
	
def CentraMoment(p,q,img):
	Mpq=0
	img = cv2.imread('image_files/obj1__0.png')
	x, y, c = img.shape
	i0=BasicMoment(1,0,img)/BasicMoment(0,0,img)
	j0= BasicMoment(0,1,img)/BasicMoment(0,0,img)
	for i in range(x):
		for j in range(y):
			Mpq=Mpq+sum(img[i,j]*pow(i-i0,p)*pow(j-j0,q))
	return Mpq

	
def NormaMoment (p,q,img):
	U=0
	Mpq=CentraMoment(p,q,img)
	gama= (p+q/2) +1
	Moo=CentraMoment(0,0,img)
	U= Mpq/math.pow(Moo,gama)
	return U
	
def hmi (img):
		HMI=[]
		HMI1= NormaMoment(0,2,img) +NormaMoment(2,0,img)
		HMI2= (NormaMoment(2,0,img)-NormaMoment(0,2,img))**2 +4*NormaMoment(1,1,img)**2  
		HMI3= (NormaMoment(3,0,img)-3*NormaMoment(1,2,img))**2 +(3*NormaMoment(2,1,img)-NormaMoment(0,3,img)**2)
		HMI4= (NormaMoment(3,0,img)+NormaMoment(1,2,img))**2 + (NormaMoment(2,1,img)+NormaMoment(0,3,img)**2)
		HMI5= (NormaMoment(3,0,img)-3*NormaMoment(1,2,img))*(NormaMoment(3,0,img)+NormaMoment(1,2,img))*((NormaMoment(3,0,img)+NormaMoment(1,2,img))**2 -3* (NormaMoment(2,1,img)+NormaMoment(0,3,img)**2))+(3*NormaMoment(2,1,img)-NormaMoment(0,3,img)*NormaMoment(2,1,img)+NormaMoment(0,3,img))*(3*(NormaMoment(3,0,img)+NormaMoment(1,2,img))**2-(NormaMoment(2,1,img)+NormaMoment(0,3,img)**2))
		HMI6=(NormaMoment(2,0,img)-NormaMoment(0,2,img))* (((NormaMoment(3,0,img)+NormaMoment(1,2,img))**2)- (NormaMoment(2,1,img)+NormaMoment(0,3,img))**2) + 4*NormaMoment(1,1,img)*(NormaMoment(3,0,img)+ NormaMoment(1,2,img))*(NormaMoment(2,1,img)+NormaMoment(0,3,img))
		HMI7= (3*NormaMoment(2,1,img)- NormaMoment(0,3,img))* (NormaMoment(3,0,img)+ NormaMoment(1,2,img))* ((NormaMoment(3,0,img)+NormaMoment(1,2,img))**2 - 3*(NormaMoment(2,1,img)+ NormaMoment(0,3,img))**2) - (NormaMoment(3,0,img)- 3*(NormaMoment(1,2,img)))* (NormaMoment(1,2,img) + NormaMoment(0,3,img))*(3*(NormaMoment(3,0,img)+ NormaMoment(1,2,img))**2 - (NormaMoment(2,1,img)+ NormaMoment(0,3,img))**2)
		HMI.append(HMI1)
		HMI.append(HMI2)
		HMI.append(HMI3)
		HMI.append(HMI4)
		HMI.append(HMI5)
		HMI.append(HMI6)
		HMI.append(HMI7)
		return HMI

def DistanceHu(img1, img2):
	dist=0
	moment1= hmi(img1)
	moment2= hmi(img2) 
	
	for i in range (len(moment1)):
		dist+= (moment1[i]-moment2[i])**2
	DisFinal= np.sqrt(dist)/7
	
	return    DisFinal
"""


### Calcul du moment du Hu en utilisant la fonction de opencv	
def MomentHU (img):
	#calul du moment de l'image 
	HU=cv2.HuMoments(cv2.moments(img)).flatten()
	return HU # return le moment de hu de l'image 
	
	
def DistanceHu(im1,im2):
	dist=0
	moment1= MomentHU(im1)# moment de l'image1
	moment2= MomentHU(im2)# moment de l'image2
	#parcourdes valeurs du moment
	for i in range (len(moment1)):
		#calcul de la distance
		dist= dist +((moment1[i]-moment2[i])**2)
		DisFinal= np.sqrt(dist)/7
	return   DisFinal # retourne la distance de HU 
#Calcul de la distance globale
def Global_fonction (histA,histB,img1,img2):
		D_glob= 0.5*distance(histA,histB)+ 0.5*DistanceHu(img1,img2)
		return D_glob # Retourne la distance globale 

		

def main():
	# variables 
	img1 = cv2.imread('image_files/obj1__0.png',0)
	rep='/media/rdius/DONNEES/IFI/INDEXATION/TP1_INDEX/TP_INDEXATION/TP_INDEX_PARTIE_1/images/'
	all_files = os.listdir(rep)
	nb_img=len(all_files)
	i=1
	histA=histogramme(rep, "obj1__0.png")
	#histB=histogramme(rep, "obj1__30.png")
	#d= distance(histA,histB)
	histB=[]
	d1=[] 
	d2=[]
	d3=[]
	k=1
	N=[]
	neighbors = []
	#img1 = cv2.imread('image_files/obj1__0.png',0)
	#moment1= hmi(img1)
	c=0
	D=0
	i=0
	img=[]
	#print img1
	D_glob=0
	#print "Distance globale avec histogrammme a 8 bits"
	#print "Calcul de distances avec les moments de HU"
	print "Calcul de la distance globale(histogramme 8 bits + moment de hu)"
	tri1 =[];tri2 =[];tri3 =[]
	
	while i <nb_img-1:
		histB=histogramme(rep, all_files[i])
		img2 = cv2.imread('images/' + all_files[i], 0)
		#fonction de calcul da la distance d'histogramme
		D_hist=distance(histA,histB)
		d1.append(D_hist)
		tri1=sorted(d1)
		D_HU=DistanceHu(img1,img2)
		d2.append(D_HU)
		tri2=sorted(d2)
		D_glob=Global_fonction(histA,histB,img1,img2)
		d3.append(D_glob)
		tri3=sorted(d3)
		
		i=i+1
		
	for i in range (len(tri1)):
		print all_files[i],"=", tri1[i] 
		
		"""
	for i in range (len(tri2)):
		print all_files[i],"=", tri2[i]
		"""
		"""
	for i in range (len(tri3)):
		print all_files[i],"=", tri3[i]
	"""	
if __name__ == "__main__":
		main()
