import minimal

g1 = minimal.MyGaussian(1.0, 0.5)

g2 = minimal.MyGaussian(0.5, 0.25)

print(dir(g1))
print(type(g1))
print(g1 + g2)