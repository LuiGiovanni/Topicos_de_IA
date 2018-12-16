#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: angel4ayala@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from CliffWalking import CliffWalking
from AgenteTD import AgenteQlearning

#entorno
entorno = CliffWalking(12,4)

# entrenamiento de agentes
agentes = 100
episodios = 500
autonomo = np.zeros(episodios)
interactive = np.zeros(episodios)

# parametros del agente autónomo
alpha = 0.5
epsilon = 0.1
gamma = 0.99

print('Comenzando entrenamiento RL...')
for r in range(agentes):
    print('Entrenando agente', r)
    entrenador = AgenteQlearning(entorno=entorno, epsilon=epsilon, alpha=alpha, gamma=gamma)
    autonomo += entrenador.entrenar(episodios)

#interactive    
#TODO: Selección de agente entrenador
feedbackProbabilidad = 0.3

print('Comenzando entrenamiento IRL...')
for r in range(agentes):    
    print('Entrenando agente', r)
    agente = AgenteQlearning(entorno=entorno, epsilon=epsilon, alpha=alpha, gamma=gamma)
    interactive += agente.entrenar(episodios, entrenador, feedbackProbabilidad)
    
#promedio de recompensa
autonomo /= agentes
interactive /= agentes

#plot
#plt.plot(autonomo, label='Autónomo')
plt.plot(interactive, label='Interactivo')
plt.xlabel('Episodios')
plt.ylabel('Recompensas durante episodios')
plt.ylim([-100, -15])
plt.legend()
plt.show()
