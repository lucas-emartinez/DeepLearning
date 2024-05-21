def computeOutput(w, x):
    z = 0.0
    
    for i in range(len(w)):
        z += x[i] * w[i]
    
    if z < 0:
        return -1
    else:
        return 1

print(computeOutput([0.9, -0.6, -0.5], [1.0, -1.0, -1.0]))
print(computeOutput([0.9, -0.6, -0.5], [1.0, -1.0, 1.0]))
print(computeOutput([0.9, -0.6, -0.5], [1.0, 1.0, -1.0]))
print(computeOutput([0.9, -0.6, -0.5], [1.0, 1.0, 1.0]))