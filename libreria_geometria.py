import sympy  as sp
import numpy as np

#Funciones auxiliares

#Función para calcular la matriz inversa
def inversa(g):
    #Calculo el determinante
    det = g.det()
    #Calculo la transpuesta
    g_transpuesta = g.transpose()
    #La paso a un array de numpy
    g_transpuesta = np.array(g_transpuesta)

    #Hago una copia para poder
    g_adjunta_de_la_transpuesta = np.array(g_transpuesta).copy()
    for i in range(len(g_transpuesta)):
        for j in range(len(g_transpuesta[i])):
            matriz_copia = g_transpuesta.copy()
            #Elimino las filas y las columnas
            sub_matriz = np.delete(matriz_copia, axis = 0, obj=i)
            sub_matriz = np.delete(sub_matriz, axis = 1, obj=j)
            #Calculo el determinante de la submatriz
            det_sub = sp.Matrix(sub_matriz).det()

            g_adjunta_de_la_transpuesta[i][j] =(-1)**(i+j)* det_sub

    g_adjunta_de_la_transpuesta = sp.Matrix(g_adjunta_de_la_transpuesta)
    #Multiplico por el inverso del determinante
    g_inv = sp.simplify( g_adjunta_de_la_transpuesta*(1/det))
    #Lo paso a un array de numpy para poder acceder a los elementos
    g_inv = np.array(g_inv)
    return g_inv
#Función para hacer la derivada parcial de los símbolos de Christoffel
def derivada_simbolos_christoffel(v_simbolos_christoffel, variables):
    matriz_diff_simbolos= [ [ [ [0 for k in range(len(variables))] for j in range(len(variables))] for i in range(len(variables))] for l in range(len(variables))]

    for v in range(len(variables)):
        for i in range(len(variables)):
            for j in range(len(variables)):
                for k in range(len(variables)):
                    if type(v_simbolos_christoffel[i][j][k]) in [ sp.core.mul.Mul ,sp.core.power.Pow,sp.core.add.Add]:
                        matriz_diff_simbolos[i][j][k][v] =sp.simplify( v_simbolos_christoffel[i][j][k].diff(variables[v]))
    return matriz_diff_simbolos

#Función para calcular la segunda derivada de la forma matricial de la métrica
def diff_g(g, variables):
    g_diff = [ [  [0for v in range(len(variables))] for j in range(len(g))] for i in range(len(g))]
    for i in range(len(g)):
        for j in range(len(g)):
            for v in range(len(variables)):
                g_diff[i][j][v]  =  g[i][j].diff(variables[v])

    g_2_diff = [ [  [ [0 for vv in range(len(variables))]for v in range(len(variables))] for j in range(len(g))] for i in range(len(g))]

    for i in range(len(g)):
        for j in range(len(g)):
            for k in range(len(variables)):
                for v in range(len(variables)):
                    g_2_diff[i][j][k][v] = g_diff[i][j][v].diff(variables[v])

    return g_2_diff


#FUNCIONES PRINCIPALES

#Función para el calculo de los simbolos de Christoffel
def simbolos_christoffel(g,variables):
    #Calculo de la inversa
    g_inv = inversa(g)

    g = np.array(g)
    #Calculo de las derivadas
    #Este vector va a contener la matriz g derivada respecto de cada la variable
    #Por ejemplo derivadas_variable[0] es la derivada de toda la matriz respecto de t
    #            derivadas_variable[1] es la derivada de toda la matriz respecto de x
    #Y derivadas_variable[1][0][0] es la derivad parcial de g_{0,0} respecto x
    # y así sucesivamente
    derivadas_variable = []

    for variable in variables:

        derivada_variable = [[0 for i in range(len(g[j]))] for j in range(len(g))]

        for i in range(len(g)):
            for j in range(len(g)):
                funcion = g[i][j]
                #Para no hacer la derivada de un número
                if type(funcion) in [ sp.core.mul.Mul ,sp.core.power.Pow,sp.core.add.Add]:
                    derivada_variable[i][j] = funcion.diff(variable)
        derivadas_variable.append(derivada_variable)
    #Calculo de los simbolos de Christoffel
    simbolos_chistoffel = [[[0 for k in variables] for j in variables] for i in variables]


    for i in range(len(variables)):
        for j in range(len(variables)):
            for k in range(len(variables)):
                simbolo = 0
                #Indice libre l
                for l in range(len(variables)):
                    if g_inv[i][l] != 0:
                        simbolo =simbolo+ g_inv[i][l]*(derivadas_variable[j][k][l]+derivadas_variable[k][j][l]-derivadas_variable[l][j][k])
                simbolos_chistoffel[i][j][k] =   sp.simplify(1/2*simbolo)

    return simbolos_chistoffel

#Función para el cálculo de las componentes del tensor de Riemann
def componentes_tensor_riemann(g, variables):
    #Calculo los simbolos de Christoffel
    v_simbolos_christoffel = simbolos_christoffel(g,variables)
    #Matriz con las derivadas parciales de los símbolos de Christoffel
    matriz_diff_simbolos = derivada_simbolos_christoffel(v_simbolos_christoffel, variables)

    # n = len(variables)
    #Matriz  n^4
    R = [ [ [ [0 for k in range(len(variables))] for j in range(len(variables))] for i in range(len(variables))] for l in range(len(variables))]

    for l in range(len(variables)):
        for i in range(len(variables)):
            for j in range(len(variables)):
                for k in range(len(variables)):
                    R[l][i][j][k] +=  v_simbolos_christoffel[l][j][k].diff(variables[i])

                    R[l][i][j][k] +=-  v_simbolos_christoffel[l][i][k].diff(variables[j])

                    #Calculo las sumas
                    s_1 = 0
                    s_2 = 0
                    #Sumatorio positivo
                    for m in range(len(variables)):
                         s_1 += v_simbolos_christoffel[m][j][k]*v_simbolos_christoffel[l][i][m]
                    #Sumatorio negativo
                    for m in range(len(variables)):
                        s_2 += v_simbolos_christoffel[m][i][k]*v_simbolos_christoffel[l][j][m]
                    R[l][i][j][k] += s_1 -s_2
                    R[l][i][j][k] = sp.simplify(R[l][i][j][k])
    return R

#Funcón para el cálculo de las componentes del tensor curvatura

def componentes_tensor_curvatura(g, variables):
    tensor_riemann = componentes_tensor_riemann(g, variables)

    tensor_curvatura= [ [ [ [0 for l in range(len(variables))] for k in range(len(variables))] for j in range(len(variables))] for i in range(len(variables))]


    g = np.array(g)

    for i in range(len(variables)):
        for j in range(len(variables)):
            for k in range(len(variables)):
                for l in range(len(variables)):

                    for m in range(len(variables)):
                        if tensor_riemann[m][i][j][k]!= 0 and g[m][l] != 0:
                            tensor_curvatura[i][j][k][l] += tensor_riemann[m][i][j][k]*g[m][l]

                    tensor_curvatura[i][j][k][l] = sp.simplify(tensor_curvatura[i][j][k][l])
    return tensor_curvatura

#Función para el calculo de las componentes del tensor de Ricci

def componentes_tensor_ricci(g,variables):
    R = componentes_tensor_riemann(g, variables)

    Ricci_curv= [ [0 for i in range(len(variables))] for j in range(len(variables))]

    for i in range(len(variables)):
        for j in range(len(variables)):
            for k in range(len(variables)):
                Ricci_curv[i][j] += R[k][k][i][j]
    return Ricci_curv

#Función para el cálculo de la curvatura escalar
def curvatura_escalar(g, variables):
    g_inv= inversa(g)
    R = componentes_tensor_ricci(g,variables)
    S = 0
    for i in range(len(variables)):
        for j in range(len(variables)):
            S += g_inv[i][j]*R[i][j]
    return sp.simplify(S)
