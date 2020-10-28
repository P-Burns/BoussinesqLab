from firedrake import *

mesh = UnitSquareMesh(10,10)
plot(mesh)

V = FunctionSpace(mesh,"CG",1)
x,y = SpatialCoordinate(mesh)
f = Function(V).interpolate(2*pi**2*sin(pi*x)*sin(pi*y))
plot(f)

v = TestFunction(V)
u = TrialFunction(V)
a = inner(grad(v),grad(u))*dx
F = inner(v,f)*dx
soln = Function(V)
boundary_ids = [1,2,3,4]
bc = DirichletBC(V,0,boundary_ids)
solve(a==F,soln,bcs=bc)

plot(soln)

exact = Function(V).interpolate(1+y)
plot(exact)


