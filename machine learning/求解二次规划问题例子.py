from cvxopt import matrix, solvers
P = matrix([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
q = matrix([0.0,0.0,0.0])
G = matrix([[-2.0,-3.0,-1.0],[-4.0,-3.0,-1.0],[0.5,0.5,1.0]])
h = matrix([-1.0,-1.0,-1.0])
# A = matrix([1.0,1.0],(1,2))#原型为cvxopt.matrix(array,dims)，等价于A = matrix([[1.0],[1.0]]）
# b = matrix([1.0])
solvers.options['show_progress'] = False
result = solvers.qp(P,q,G,h)

print('x=',result['x'].T)

print('status=',result['status'])
print('gap=',result['gap'])
print('relative gap=',result['relative gap'])
print('primal objective=',result['primal objective'])
print('dual objective=',result['dual objective'])