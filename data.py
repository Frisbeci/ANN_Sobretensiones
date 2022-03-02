import sys
sys.path.insert(0, './LisRead')
import LISread_det
import pandas as pd
import numpy as np

def max_abs(vec):
    return max([abs(vec[0]), abs(vec[1]), abs(vec[2]), abs(vec[3]), abs(vec[4]), abs(vec[5])])

BASE_LN = 500000*np.sqrt(2)/np.sqrt(3)
BASE_LL = 500000*np.sqrt(2)

def read_data(path, file, n_sim):
    r = LISread_det.read_det(f'{path}{file}{1}.lis', lineSplit='\n')
    chn = r['channels']  # se definen los nombres de las variables para buscar las de utilidad
    #col = chn[chn.index('mLONG'):]
    #for c in range(len(col)):
    #    col[c] = col[c][1:]
    car=['LONG','RDIVX','X0F','X1F','RINSER','CLOSE1','RPPOS','RPCERO','LPPOS','LPCERO','CPPOS','CPCERO','SIMET','ENC1','ENC2','ENC1AS','ENC2AS']
    calc=['COM1','COM2']
    col=np.append(car,calc)
    col = np.append(col, ['Vmax_fn_EMISOR', 'Vmax_ff_EMISOR', 'Vmax_fn_RECEPTOR', 'Vmax_ff_RECEPTOR'])
    data = pd.DataFrame(columns=col)

    for i in range(1, n_sim):  # se recorren las 1000 simulaciones
        try:
            r = LISread_det.read_det(f'{path}{file}{i}.lis', lineSplit='\n')  # se extraen los datos de una simulación en particular
            row = np.array([])
            for j in range(len(car)):
                row = np.append(row, r['vmax'][:, chn.index(f'm{car[j]}')][0])

            if row[-5] == 1: #se calcula la variable binaria de la compensacion simetrica o asimetrica
                com = [row[-4],row[-3]]
            else:
                com = [row[-2],row[-1]]

            row = np.append(row, com)

            o1 = [r['vmax'][:, chn.index('vLN1A-')][0]/BASE_LN, r['vmax'][:, chn.index('vLN1B-')][0]/BASE_LN, r['vmax'][:, chn.index('vLN1C-')][0]/BASE_LN, r['vmin'][:, chn.index('vLN1A-')][0]/BASE_LN, r['vmin'][:, chn.index('vLN1B-')][0]/BASE_LN, r['vmin'][:, chn.index('vLN1C-')][0]/BASE_LN]
            o2 = [r['vmax'][:, chn.index('vLN1A-LN1B')][0]/BASE_LL, r['vmax'][:, chn.index('vLN1B-LN1C')][0]/BASE_LL, r['vmax'][:, chn.index('vLN1C-LN1A')][0]/BASE_LL, r['vmin'][:, chn.index('vLN1A-LN1B')][0]/BASE_LL, r['vmin'][:, chn.index('vLN1B-LN1C')][0]/BASE_LL, r['vmin'][:, chn.index('vLN1C-LN1A')][0]/BASE_LL]
            o3 = [r['vmax'][:, chn.index('vVFF2A-')][0]/BASE_LN, r['vmax'][:, chn.index('vVFF2B-')][0]/BASE_LN, r['vmax'][:, chn.index('vVFF2C-')][0]/BASE_LN, r['vmin'][:, chn.index('vVFF2A-')][0]/BASE_LN, r['vmin'][:, chn.index('vVFF2B-')][0]/BASE_LN, r['vmin'][:, chn.index('vVFF2C-')][0]/BASE_LN]
            o4 = [r['vmax'][:, chn.index('vVFF2A-VFF2B')][0]/BASE_LL, r['vmax'][:, chn.index('vVFF2B-VFF2C')][0]/BASE_LL, r['vmax'][:, chn.index('vVFF2C-VFF2A')][0]/BASE_LL, r['vmin'][:, chn.index('vVFF2A-VFF2B')][0]/BASE_LL, r['vmin'][:, chn.index('vVFF2B-VFF2C')][0]/BASE_LL, r['vmin'][:, chn.index('vVFF2C-VFF2A')][0]/BASE_LL]

            row = np.append(row, [max_abs(o1), max_abs(o2), max_abs(o3), max_abs(o4)])
            data.loc[len(data)] = row
        except:
            print(f'Simulación n°{i} con errores')

    return data.drop(columns=['ENC1','ENC2','ENC1AS','ENC2AS'])