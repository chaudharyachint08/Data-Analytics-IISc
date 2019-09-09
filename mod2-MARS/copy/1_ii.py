import math
import numpy as np
from scipy.optimize import minimize
import scipy

r=np.zeros(13)
b=[13]
c=[13]

#r=[[0 for w in range(13)]
a=[[0 for p in range(13)] for q in range(13)] 
#


def objfun(x0):
	x=x0[0]
	y=x0[1]
	gm=1
	t=0
	for t in range(1,13):
		alpha=math.radians(a[t][3]*30+a[t][4]+a[t][5]/60+a[t][6]/3600)
		beta=math.radians(a[t][9]*30+a[t][10]+a[t][11]/60+a[t][12]/3600)
		d1=1/math.tan(alpha-y)
		d2=1/math.tan(beta-y)
		if d1>d2:
			d3=d1-d2
		else:
			d3=d2-d1
		r[t]=math.sqrt(math.pow(x,2)*(1+math.pow(d1,2))+2*x*(1+d1*d2)+1+math.pow(d2,2))/d3
	am=np.mean(r)
	for i in range(1,13):
		gm=gm*r[i]
	gm=math.pow(gm,1/12)
	res=math.log(am)-math.log(gm)
	return res



f = open("data/01_data_mars_opposition.csv", "rt")

i=0


#a=[[0 for x in range(13)] for y in range(13)] 


b=f.read().splitlines()


for i in range(1,13):
	c=b[i].split(",")
	for j in range(len(c)):
		a[i][j]=int(c[j])


bounds=((0,280),(0,4))
x_in=[8000,1000]
y0=scipy.optimize.minimize(objfun,x_in,bounds=bounds,method='SLSQP')
print(y0)










