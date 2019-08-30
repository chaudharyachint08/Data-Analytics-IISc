import numpy as np
import matplotlib.pyplot as plt

p1, p2 = 500,500
X, Y1, Y2 = 50, 100,150

x1 = np.linspace(0,X,p1)
x2 = np.linspace(0,X,p1)

y1 = np.random.random(p1)*Y1*(x1/x1.max())
y2 = np.random.random(p2)*Y2*(x2/x2.max())

plt.scatter(x1,y1,s=8)
plt.scatter(x2,y2,s=8)

plt.xlabel('Overs Remaining')
plt.ylabel('Runs Scored')
	

plt.show()
