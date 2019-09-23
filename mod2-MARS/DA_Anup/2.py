import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from scipy.spatial import distance


# ### Input File Read 
data=pd.read_csv("./../data/01_data_mars_triangulation.csv")

# ### Sun Longitude 
earth_heliocentric_longitude=data.iloc[:,4:6]

#earth_heliocentric_longitude
mars_geocentric_longitude=data.iloc[:,6:8]

#mars_geocentric_longitude
degree=data["DegreeEarthLocationHelioCentric"].values
minute=data["MinuteEarthLocationHelioCentric"].values

earth_heliocentric_longitude_radian=(np.pi)/180 * (degree + (minute/60))
#earth_heliocentric_longitude_radian
degree_mars=data["DegreeMarsLocationGeoCentric"].values
minute_mars=data["MinuteMarsLocationGeoCentric"].values

mars_geocentric_longitude_radian= (np.pi)/180 * (degree_mars + (minute_mars/60))

#mars_geocentric_longitude_radian

# ### Optimization 
theta=earth_heliocentric_longitude_radian
phi=mars_geocentric_longitude_radian

#theta
#phi
#Question (i)

def projection(theta,phi):
    x_list=[]
    y_list=[]
    r_list=[]
    theta_list=[]
    for i in range(0,len(theta),2):
        x=(np.sin(theta[i+1])-np.sin(theta[i])) + ((np.tan(phi[i])*np.cos(theta[i])) - (np.tan(phi[i+1])*np.cos(theta[i+1])))
        x_mars=x/(np.tan(phi[i]) - np.tan(phi[i+1]))
        y_mars= np.tan(phi[i])*x_mars + (np.sin(theta[i]) - np.tan(phi[i])*np.cos(theta[i]))
        theta_mars=np.arctan(y_mars/x_mars)
        r=np.sqrt(x_mars**2 + y_mars**2)
        x_list.append(x_mars)
        y_list.append(y_mars)
        r_list.append(r)
        #print(y_mars/x_mars)
        tmp=theta_mars*(180/np.pi)
        theta_list.append(tmp)
        #print(i)

    return r_list,theta_list


# In[18]:


r_list,projection=projection(theta,phi)


#projection
#r_list

def loss_circle(params,args):
    x_list=[]
    y_list=[]
    r_list=[]
    r_theta=[]
    theta=args[0]
    phi=args[1]
    radius=params[0]
    loss=0
    for i in range(0,len(theta),2):
        x=(np.sin(theta[i+1])-np.sin(theta[i])) + ((np.tan(phi[i])*np.cos(theta[i])) - (np.tan(phi[i+1])*np.cos(theta[i+1])))
        x_mars=x/(np.tan(phi[i]) - np.tan(phi[i+1]))
        y_mars= np.tan(phi[i])*x_mars + (np.sin(theta[i]) - np.tan(phi[i])*np.cos(theta[i]))
        r=np.sqrt(x_mars**2 + y_mars**2)
        x_list.append(x_mars)
        y_list.append(y_mars)
        r_list.append(r)
    for i in range(5):
        loss+=(radius-r_list[i])**2
        #print("loss")
        
    #print(r_list)
    #print(r_theta)

    #print((math.log(ap,10) - math.log(gp,10)))
    return loss


def optimizer_circle_global(function,method_name,theta,phi):
    
    from scipy.optimize import basinhopping

    #radius=[1]
    
    initial_parameters = [1] #Take any value
    #bound to avoid case of global Minima where i am getting Loss = 0
    #bounds = [(0.1, np.inf) for _ in a] + [(-np.inf, np.inf)]
    
    minimizer_kwargs = {"method":method_name, "args":[theta,phi]}
    parameters = basinhopping(function, initial_parameters,
                      minimizer_kwargs=minimizer_kwargs)

    #optimized_params, loss = parameters['x'], parameters['fun']
    #print(optimized_params1)
    #print(squared_error_loss1)
    return parameters['x'], parameters['fun']


# ### Optimize
#Global Minimizer
from scipy.spatial import distance
print("Optimizing Parameters .... ")
function_name=loss_circle
optimized_params, loss= optimizer_circle_global(function_name,'BFGS',theta,phi)
print("Optimized Parameters Computed")


#optimized_params
#Radius = 1.57732091

#loss

x_list=[]
y_list=[]
r_list=[]
r_theta=[]
radius=optimized_params[0]
for i in range(0,len(theta),2):
        x=(np.sin(theta[i+1])-np.sin(theta[i])) + ((np.tan(phi[i])*np.cos(theta[i])) - (np.tan(phi[i+1])*np.cos(theta[i+1])))
        x_mars=x/(np.tan(phi[i]) - np.tan(phi[i+1]))
        y_mars= np.tan(phi[i])*x_mars + (np.sin(theta[i]) - np.tan(phi[i])*np.cos(theta[i]))
        r=np.sqrt(x_mars**2 + y_mars**2)
        x_list.append(x_mars)
        y_list.append(y_mars)
        r_list.append(r)
#print(r_theta)


import matplotlib.pyplot as plt
#plt.figure(figsize=(10,10))
circle1 = plt.Circle((0, 0), 1.58,fill=False)
fig, ax = plt.subplots(figsize=(5, 5))
ax.add_artist(circle1)
plt.scatter(x_list,y_list,color='red')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()

