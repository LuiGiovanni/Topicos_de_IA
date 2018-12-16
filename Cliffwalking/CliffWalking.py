#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class CliffWalking(object):
    
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
        self.agentPos = [0, 0]
        #acciones
        self.accion_arr = 0
        self.accion_aba = 1
        self.accion_izq = 2
        self.accion_der = 3
        self.acciones = [self.accion_arr, self.accion_aba, self.accion_izq, self.accion_der]
        # zonas
        self.startPos = [0, alto -1]
        self.goalPos = [ancho -1, alto -1]
    # end __init__
    
    def reset(self):
        self.agentPos = self.startPos
        return self.agentPos
    # end reset
    
    def actuar(self, accion):
        x, y = self.agentPos
        
        if (accion == 0): # arriba
            y = max(y -1, 0)
        elif (accion == 1): # abajo
            y = min(y +1, self.alto -1)
        elif (accion == 2): # izquierda
            x = max(x -1, 0)
        elif (accion == 3): # derecha
            x = min(x +1, self.ancho -1)
        else:
            print('Error, accion desconocida')
            assert False
            
        estado = [x, y]
        reward = -1
        self.agentPos = estado
            
        if (accion == self.accion_aba and y == 2 and 1 <= x <= 10) or (
            accion == self.accion_der and self.agentPos == self.startPos):
            reward = -100
            self.reset()
        
        return self.agentPos, reward
    # end actuar
