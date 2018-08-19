import numpy as np 

# X = (horas dormindo, horas estudando), y = pontuação no teste
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# unidades de escala
x = x/np.amax(x, axis=0) # máximo no array x
y = y/100 # pontuação máxima do teste é de 100

class Neural_Network(object):
    def __init__ (self):
        # parâmetros
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        # pesos
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize) #(3x2) matriz de peso da entrada para a camada oculta
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize) #(3x1) matriz de peso da camada oculta para a de saída  

    def forward (self, x):
        #propagação para a frente através da nossa rede
        self.z = np.dot(x, self.w1) #produto ponto de X (entrada) e primeiro conjunto de pesos 3x2
        self.z2 = self.sigmoid(self.z) # função de ativação
        self.z3 = np.dot(self.z2, self.w2) # produto de ponto da camada oculta (z2) e segundo conjunto de pesos 3x1
        o = self.sigmoid(self.z3) # função de ativação final
        return o

    def sigmoid (self, s):
        # função de ativação
        return 1/(1 + np.exp(-s))

    def sigmoidPrime (self, s):
        # derivado do sigmoid
        return s * (1 - s)  

    def backward (self, x, y, o):
        # propagação para trás através da rede
        self.o_error = y - o # erro na saída
        self.o_delta = self.o_error * self.sigmoidPrime(o) #aplicando derivado de sigmóid ao erro
        self.z2_error = self.o_delta.dot(self.w2.T) #z2_error: quanto nossos pesos de camada oculta contribuíram para o erro de saída
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) #aplicando derivado do sigmóid no z2_error
        self.w1 += x.T.dot(self.z2_delta) #ajustando o primeiro conjunto (entrada -> oculto) pesos
        self.w2 += self.z2.T.dot(self.o_delta) #Ajustando os pesos do segundo conjunto (oculto -> saída)

    def train (self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)  

    def predict (self):
        print("Dados previstos com base em pesos treinados: ")
        print("Entrada (escalada):   \n" + str(xPredicted))
        print("Saída: \n" + str(self.forward(xPredicted)))

    def saveWeights (self):
        np.savetxt("w1.txt", self.w1, fmt="%s")
        np.savetxt("w2.txt", self.w2, fmt="%s")
          

NN = Neural_Network()

for i in range(1000): #treina a NN 1.000 vezes
    print("Entrada: \n" + str(x))
    print("Saída Real: \n" + str(y))
    print("Saída Prevista: \n" + str(NN.forward(x)))
    print("Perda: \n" + str(np.mean(np.square(y - NN.forward(x))))) #soma  a media quadrada da perda
    print("\n")
    NN.train(x, y)

#definindo nossa saída
"""o = NN.forward(x)

print("Saída Prevista: \n" + str(o))
print("Saída Real: \n" + str(y))"""

xPredicted = np.array(([4,8]), dtype=float)

xPredicted = xPredicted/np.amax(xPredicted, axis=0) # máximo da nossa variável (nossos dados de entrada para a previsão)

NN.saveWeights()
NN.predict()







