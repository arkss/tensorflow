import numpy as np

data = np.array([[1,2,3],[4,5,6],[7,8,9]])


numerator = data - np.min(data, 0)
denominator = np.max(data, 0) - np.min(data, 0)
a = numerator / (denominator + 1e-7)

print(data,"\n")
print(np.min(data, 0),"\n")
print(numerator,"\n")
print(np.max(data, 0),"\n")
print(denominator,"\n")