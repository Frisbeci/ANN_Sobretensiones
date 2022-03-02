#
# leer un LIS de simulacion 
#

import re
import numpy as np


def read(fname, onlyExtrema=True,lineSplit='\n'):
    """procesar un archivo .LIS generado por ATP para obtener los valores maximos
y minimos y tiempos de switcheo durante simulaciones estadisticas.
A la salida devuelve un diccionario con:
    - lista 'channels' de nombres de canales (usa vDESDE-HASTA para las tensiones
      iDESDE-HASTA para las corrientes, pDESDE-HASTA para las potencias,
      eDESDE-HASTA para las energias, mMNODE para los valores desde MODELS,
      tTNODE para los valores desde TACS.
      
    - diccionario de switches {numero: (nodoDesde, nodoHasta),...}
    
    - lista 'sw' de numeros de switches correspondientes a los tiempos
    - np.array 'tsw' con los tiempo de switcheo de los switches monitoreados
    - np.array 'vext' con los valores extremos de las variables monitoreadas
    - np.array 'text' con los tiempos correspondientes.
    
    - np.array 'vmax' con los valores maximos de las variables monitoreadas
    - np.array 'vmin' con los valores minimos de las variables monitoreadas
    - np.array 'tmax' con los tiempos en los que se produce el 'vmax'
    - np.array 'tmin' con los tiempos en los que se produce el 'vmin'

Si onlyExtrema es False, genera los arrays de vmax, vmin, tmax, tmin, ademas de
los valores extremos.
"""
    #
    # primero, lo primero, abrir el LIS y leerlo.
    #
    r = open(fname,'r').read()
    
    #
    # comprobar que exista la frase "Random switching times for simulation number"
    #
    if not "Random switching times for simulation number" in r:
        raise Exception("Solamente se pueden procesar archivos LIS producidos por simulaciones estadisticas")
    
    #
    # Separar el LIS con "Random switching times for simulation number" como clave ...
    #
    rs = re.split("Random switching times for simulation number", r)
    ###print rs[1]
    
    # rs es ahora una lista de cadenas, conteniendo lo que hay desde la primer linea del LIS
    # hasta la primer ocurrencia de la frase "Random switching times for simulation number"
    # y luego una secuencia de varias cadenas, con la informacion de las distintas energizaciones
    #
    
    # rs[0] es la primer parte del archivo LIS, que contiene entre un monton de
    # cosas el ATP que finalmente utilizo para hacer la simulacion.
    # El contenido del archivo ATP esta en la primera parte del LIS luego de una barra | 
    atp = re.findall(".+\|(?P<atp>.+)", rs[0].replace('\r',''))
    # en atp, se tiene renglon por renglon el archivo ATP (mas alguna basura adicional)
    n1 = atp.index('BLANK BRANCH') # aca terminan las ramas
    n2 = atp.index('BLANK SWITCH') # aca terminan los switches
    # obtener un diccionario de switches del modelo {numeroDeSwitch: (nodoDesde, nodoHasta)}
    ns = 1
    switches = {}
    for k in range(n1+1,n2):
        if not atp[k].startswith('C'):
            f = atp[k][2:8].strip() # entre las columnas 2 y 7 (inclusive)
            t = atp[k][9:15].strip() # entre las columnas 9 y 14 (inclusive)
            switches[ns] = (f, t) # from - to
            ns += 1 # contar un switch mas
    
    #
    # Buscar la lista de canales representados en los reportes de maxima / minima
    #
    f = rs[0].split(lineSplit) # separar la primer seccion renglon-por-renglon
    
    kk = -1
    for k in range(len(f)):
        if f[k].startswith('Column headings for the'):
            kk = k # ubicar en que renglon comienza con la informacion de canales
            break
    #
    # Obtener el numero de canales de simulacion almacenados...
    #
    m = re.match("[A-Za-z ]+(?P<nv>[0-9]+)[ \w]+",f[kk])
    nv = int(m.group('nv'))
    kk += 1 # avanza al siguiente renglon
        
    # en el LIS, los renglones siguientes indican de que tipo son los canales
    # almacenados (pueden ser tension, corriente, potencia, energia, TACS o MODELS 
    types = []
    nrem = nv
    while nrem > 0: # leer tantos renglones como sea necesario, hasta agotar la cantidad de canales grabados
        m = re.match("\s+(First|Next)\s+(?P<ng>[0-9]+)\s+output variables (are|belong to) (?P<type>[^(]+)",f[kk])
        if m:
            ng = int(m.group('ng'))
            nrem -= ng
            kk += 1
            types.append((ng, m.group('type').strip()))
        else:
            # paso algo no esperado
            raise Exception("una situacion no esperada se ha presentado cuando estaba leyendo la lista de canales ... rajemos !")
    
    # si el siguiente renglon inicia con "Branch power" o "Branch energy"
    npow = 0
    if f[kk].startswith('Branch power'):
        m = re.match(".+\((?P<pow>[0-9]+)", f[kk])
        npow = int(m.group('pow'))
        kk += 1 # avanza al renglon siguiente
    
    # o Branch energy
    nenerg = 0
    if f[kk].startswith('Branch enery'):
        m = re.match(".+\((?P<energ>[0-9]+)", f[kk])
        nenerg = int(m.group('energ'))
        kk += 1 # avanza al renglon siguiente
    
    #
    # Ahora, procesar la lista de nombres de nodos (DESDE - HASTA) que corresponden
    # con los nv canales en total.
    #
    nvars = []
    nrem = nv
    while nrem > 0:    
        l1 = []
        for i in range(17,len(f[kk]),12):
            l1.append(f[kk][i:i+12].strip())
        l2 = []
        for i in range(17,len(f[kk+1]),12):
            l2.append(f[kk+1][i:i+12].strip())
        
        ll = zip(l1, l2)
        nvars.extend(ll)
        nrem -= len(ll) # descuenta del total, los que leyo en este par de renglones
        kk += 3 # avanza 3 renglones (hay un renglon vacio entre medio
    
    #
    # Finalmente, construir una lista de nombres con una inicial v, i, p, e, m o t
    # dependiendo del tipo de senial registrada.
    #
    nameVars = []
    k = 0
    for n,ntype in types:
        if ntype.startswith('electric'):
            # tensiones o potencias
            # npow potencias
            # n-npow tensiones, en ese orden
            while n > 0:
                if npow > 0:
                    nameVars.append( 'p%s-%s' % nvars[k] )
                    npow -= 1
                else:
                    nameVars.append( 'v%s-%s' % nvars[k] )
                n -= 1
                k += 1 # avanza al siguiente par de nombres.
        
        elif ntype.startswith('branch currents'):
            # corrientes o energia
            # nenerg energias
            # n - nenerg corrientes
            while n > 0:
                if nenerg > 0:
                    nameVars.append( 'e%s-%s' % nvars[k] )
                    nenerg -= 1
                else:
                    nameVars.append( 'i%s-%s' % nvars[k] )
                n -= 1
                k += 1 # avanza al siguiente par de nombres
        
        elif ntype.startswith('MODELS'):
            while n > 0:
                nameVars.append( 'm%s' % nvars[k][1] )
                n -= 1
                k += 1 # avanza al siguiente par de nombres
        
        elif ntype.startswith('TACS'):
            while n > 0:
                nameVars.append( 't%s' % nvars[k][1] )
                n -= 1
                k += 1 # avanza al siguiente par de nombres
    
    #    
    # el numero de simulaciones que se realizaron es len(rs) - 1
    #
    n = len(rs) - 1 # numero de simulaciones
    
    #
    # des-empaquetar la informacion de las corridas.
    #
    res = {'sw':[], 'tsw':[], 'vmax':[], 'vmin':[], 'tmax':[], 'tmin':[], 'vext':[], 'text':[]}
    for k in range(1,n+1):
        w = rs[k].replace('%s%s'%(lineSplit,lineSplit),'\nVariable extrema : ') # separa con dos \r\n consecutivos ....
        # obtener las listas de valores
        if not onlyExtrema:
            m = re.match(r"\s+(?P<num>[0-9]+)\s+:\s+(?P<bts>([0-9]+\s+[0-9\.\-E]+\s+)+)"\
                ".+"\
                "Variable maxima :\s+(?P<vmax>([0-9\.\-E]+\s*)+)"\
                "Times of maxima :\s+(?P<tmax>([0-9\.\-E]+\s*)+)"\
                "Variable minima :\s+(?P<vmin>([0-9\.\-E]+\s*)+)"\
                "Times of minima :\s+(?P<tmin>([0-9\.\-E]+\s*)+)"\
                "Variable extrema :\s+(?P<extrema>([0-9\.\-E]+\s*)+)"\
                "Times of maxima :\s+(?P<textrema>([0-9\.\-E]+\s*)+)"\
                , w, re.MULTILINE|re.DOTALL)
        else:
            m = re.match(r"\s+(?P<num>[0-9]+)\s+:\s+(?P<bts>([0-9]+\s+[0-9\.\-E]+\s+)+)"\
                ".+"\
                "Variable extrema :\s+(?P<extrema>([0-9\.\-E]+\s*)+)"\
                "Times of maxima :\s+(?P<textrema>([0-9\.\-E]+\s*)+)"\
                , w, re.MULTILINE|re.DOTALL)
        
        if not m:
            if not onlyExtrema:
                raise Exception("Una excepcion ha ocurrido al procesar los registros.\nUsar onlyExtrema=True si el archivo LIS no tiene los valores minimos y maximos")

            raise Exception("Ha ocurrido una excepcion procesando los registros...")
        
        res['vext'].append( map(float,m.group('extrema').replace(lineSplit,'').split()) )
        res['text'].append( map(float,m.group('textrema').replace(lineSplit,'').split()) )
        if not onlyExtrema:
            res['vmax'].append( map(float,m.group('vmax').replace(lineSplit,'').split()) )
            res['tmax'].append( map(float,m.group('tmax').replace(lineSplit,'').split()) )
            res['vmin'].append( map(float,m.group('vmin').replace(lineSplit,'').split()) )
            res['tmin'].append( map(float,m.group('tmin').replace(lineSplit,'').split()) )
        
        # procesar la lista de interruptores y sus tiempos de actuacion
        bts = m.group('bts').replace(lineSplit,'').split()
        b = map(int, bts[0::2]) # numero de switch, compatible con el diccionario 'switches'
        t = map(float, bts[1::2])
        
        if not res['sw']:
            res['sw'] = b
        res['tsw'].append(t)
    
    #
    # convertir todas las listas de canales en np.array para su manipulacion
    #   
    res['vext'] = np.array(res['vext'])
    res['text'] = np.array(res['text'])
    if not onlyExtrema:
        res['vmax'] = np.array(res['vmax'])
        res['tmax'] = np.array(res['tmax'])
        res['vmin'] = np.array(res['vmin'])
        res['tmin'] = np.array(res['tmin'])
    res['tsw'] = np.array(res['tsw'])
    res['switches'] = switches # la lista de switches (completa)
    res['channels'] = nameVars # nombres de canales de simulacion
    
    return res 


if __name__ == '__main__':
    r = read('prueba6interruptor-estadistico_minmax.lis',onlyExtrema=True)
    #r = read('IexMax-1000ptsmm.LIS')
    

