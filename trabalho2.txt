import numpy as np

#definindo o sistema de equacoes nao lineares
def F(x):
    f1 = 3*x[0] + np.cos(x[1]*x[2]) - 0.5
    f2 = x[0]**2 - 81*(x[1] + 0.1)**2 + np.sin(x[2]) + 1.06
    f3 = np.exp(-x[0]*x[1]) + 20*x[2] + (10*np.pi - 3)/3
    return np.array([f1, f2, f3])

#definindo a jacobiana
def J(x):
    j11 = 3
    j12 = -x[2] * np.sin(x[1]*x[2])
    j13 = -x[1] * np.sin(x[1]*x[2]) #primeira expressao (f1)
    
    j21 = 2*x[0] 
    j22 = -162*(x[1] + 0.1)
    j23 = np.cos(x[2])              #segunda expressao(f2) 
    
    j31 = -x[1]*np.exp(-x[0]*x[1])
    j32 = -x[0]*np.exp(-x[0]*x[1])  
    j33 = 20                        #terceira expressao(f3)
    
    return np.array([
        [j11, j12, j13],
        [j21, j22, j23],
        [j31, j32, j33]
    ])                              #montando a matriz das derivadas

#metodo de newton
def newton_system(F, J, x0, tol=1e-6, max_iter=50): #F = sistema nao linear, J = jacobiana, x0 = chute inicial, tol = tolerancia 10^-6 e max_iter = maximo de iteracoes
    x = x0                                      
    for i in range(max_iter):
        Fx = F(x)             #recebe F no ponto atual x (ponto no loop)
        Jx = J(x)             #recebe J no ponto atual x (ponto no loop)

        delta = np.linalg.solve(Jx, -Fx) #resolve o sistema linear J * delta = -F para encontrar delta

        x = x + delta         #atualiza a estimativa da solução

        if np.linalg.norm(delta, ord=2) < tol:  #faz a norma com o criterio de parada < que a tol = 10^-6
            print(f"Convergiu em {i+1} iterações.") 
            return x  #retorna o x

    print("Não convergiu dentro do número máximo de iterações.")
    return x 

x0 = np.array([0.1, 0.1, -0.1])  #chute inicial
sol = newton_system(F, J, x0)    #chama a funcao

print("Solução aproximada:", sol)
