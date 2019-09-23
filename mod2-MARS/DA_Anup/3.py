import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean


# ### Input File Read 
data=pd.read_csv("./../data/01_data_mars_opposition.csv")


mars_heliocentric_longitude=data.iloc[:,3:7]


s=data["ZodiacIndex"].values
degree=data["Degree"].values
minute=data["Minute"].values
seconds=data["Second"].values


mars_heliocentric_longitude_in_degree= s*30 +degree + (minute/60) + (seconds/3600)


#mars_heliocentric_longitude_in_degree
mars_heliocentric_longitude_in_radian= mars_heliocentric_longitude_in_degree*math.pi/180.0

#mars_heliocentric_longitude_in_radian
#geocentric_latitude=data.iloc[:,7:9]

#geocentric_latitude.head(12)

#Not Required for First Part
geocentric_latitude_in_radian=(np.pi/180)*((geocentric_latitude["LatDegree"].values )+ (geocentric_latitude["LatMinute"].values /60))

#geocentric_latitude_in_radian

#mars_heliocentric_longitude_in_radian


# # Q3 _1
orbit_radius=1.57 #Computed in Part 2

mars_heliocentric_latitude=np.arctan(np.tan(geocentric_latitude_in_radian)*(1-(1/orbit_radius)))

#np.mean(mars_heliocentric_latitude*(180/np.pi))


# # Q3_2

x_list=[]
y_list=[]
z_list=[]
for i in range(len(data)):
    x_list.append(orbit_radius*np.cos(mars_heliocentric_longitude_in_radian[i])*np.cos(mars_heliocentric_latitude[i]))
    y_list.append(orbit_radius*np.sin(mars_heliocentric_longitude_in_radian[i])*np.cos(mars_heliocentric_latitude[i]))
    z_list.append(orbit_radius*np.sin(mars_heliocentric_latitude[i]))
    



# ### Inclination of Mars orbit to Reference Axis 
def loss_function(params,args):
    
    a=params[0]
    b=params[1]
    c=params[2]
    d=params[3]
    
    x=args[0]
    y=args[1]
    z=args[2]
    loss=0
    for i in range(len(y)):
        d = abs((a * x[i] + b * y[i] + c * z[i] + d))  
        e = (math.sqrt(a * a + b * b + c * c))
        loss+=(d/e)
        
    
    #print(loss)
    return loss


def optimizer_global(function,method_name,x_list,y_list,z_list):
    
    from scipy.optimize import basinhopping

    initial_parameters = [10,10,10,10] #You can take any parameter [Global Minmizer]
    
    #bound to avoid case of global Minima where i am getting Loss = 0
    #bounds = [(0.1, np.inf) for _ in a] + [(-np.inf, np.inf)]
    minimizer_kwargs = {"method":method_name, "args":[x_list,y_list,z_list]}
    parameters = basinhopping(function, initial_parameters,
                      minimizer_kwargs=minimizer_kwargs)
    #optimized_params, loss = parameters['x'], parameters['fun']
    #print(optimized_params1)
    #print(squared_error_loss1)
    return parameters['x'], parameters['fun']


# In[263]:


#Global Minimizer
print("Optimizing Parameters .... ")
function_name=loss_function
optimized_params, loss= optimizer_global(function_name,'BFGS',x_list,y_list,z_list)
print("Optimized Parameters Computed")



#optimized_params
inclination=np.arccos(optimized_params[2]/np.linalg.norm(optimized_params))*(180/np.pi)


print("Inclination= " +repr(round(inclination,2)) + " degrees") 
#Inclination= 1.83 Degrees