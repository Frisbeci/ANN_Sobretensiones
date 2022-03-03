# !/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np # Python 3.7, tensorflow 2.0, scikit-learn; importar automaticamente from tensorflow.keras.models import load_model y from joblib import load ;;;;;;;;  python -m http.server --bind localhost --cgi 8000
from tensorflow.keras.models import load_model
from joblib import load
import cgi
import cgitb
import matplotlib.pyplot as plt
import random

def plot_diagrama(Uff_emi_pu,Ufn_emi_pu,Uff_rec_pu,Ufn_rec_pu):
    base = 500*np.sqrt(2)
    Uff_emi_kv = base*Uff_emi_pu
    Ufn_emi_kv = base*Ufn_emi_pu/np.sqrt(3)
    Uff_rec_kv = base*Uff_rec_pu
    Ufn_rec_kv = base*Ufn_rec_pu/np.sqrt(3)
    maxpu = 3.5
    path ='utils/'
    fname ='sistema.png'
    im = plt.imread(path+fname)
    plt.imshow(im)
    plt.text(300,150,'Lado Emisor')
    plt.text(150,250,'Uf-f = '+str(Uff_emi_pu)+' [pu] | '+str(int(Uff_emi_kv))+' [kV]', color = [min(Uff_emi_pu/maxpu,1),1-min(Uff_emi_pu/maxpu,1),0],fontweight = 'bold')
    plt.text(150,350,'Uf-n = '+str(Ufn_emi_pu)+' [pu] | '+str(int(Ufn_emi_kv))+' [kV]', color = [min(Ufn_emi_pu/maxpu,1),1-min(Ufn_emi_pu/maxpu,1),0],fontweight = 'bold')
    plt.text(1350,150,'Lado Receptor')
    plt.text(1200,250,'Uf-f = '+str(Uff_rec_pu)+' [pu] | '+str(int(Uff_rec_kv))+' [kV]', color = [min(Uff_rec_pu/maxpu,1),1-min(Uff_rec_pu/maxpu,1),0],fontweight = 'bold')
    plt.text(1200,350,'Uf-n = '+str(Ufn_rec_pu)+' [pu] | '+str(int(Ufn_rec_kv))+' [kV]', color = [min(Ufn_rec_pu/maxpu,1),1-min(Ufn_rec_pu/maxpu,1),0],fontweight = 'bold')
    plt.axis('off')
    plt.savefig('utils/referencia.png')

loaded_model = load_model('red_entrenada')
scaler_x = load('escalador/MinMaxScaler_x')
scaler_y = load('escalador/MinMaxScaler_y')

def sobretensiones(vector):
    vector_np = np.array([vector,])
    vector_scale = scaler_x.transform(vector_np)
    vector_pred = loaded_model.predict(vector_scale)
    return {'Predicciones': scaler_y.inverse_transform(vector_pred)[0]}

cgitb.enable()
form = cgi.FieldStorage()

o1=[] #arreglos vacios para realizar el barrido estadistico de cierre
o2=[]
o3=[]
o4=[]

barrido = np.linspace(0.04, 0.06, 100)
for close1 in barrido:
    vector_c = [float(form['long'].value), float(form['rdivx'].value) / 100.0,
                (np.sqrt(3) * 525000 / float(form['icc1f'].value) - (2 * 525000 / (np.sqrt(3) * float(form['icc3f'].value)))),
                525000 / (np.sqrt(3) * float(form['icc3f'].value)), float(form['rinser'].value), close1,
                float(form['rppos'].value) / 10000.0, float(form['rpcero'].value) / 1000.0,
                float(form['lppos'].value) / 1000.0,
                float(form['lpcero'].value) / 100.0, float(form['cppos'].value) / 10000.0,
                float(form['cpcero'].value) / 10000.0,
                float(form['simet'].value), float(form['com1'].value), float(form['com2'].value)]
    calculos = sobretensiones(vector_c)

    o1 = np.append(o1,round(float(calculos['Predicciones'][0]), 4)) #sqrt(3)
    o2 = np.append(o2,round(float(calculos['Predicciones'][1]), 4))
    o3 = np.append(o3,round(float(calculos['Predicciones'][2]), 4)) #sqrt(3)
    o4 = np.append(o4,round(float(calculos['Predicciones'][3]), 4))


o1_98 = round(np.mean(o1)+2.06*np.std(o1),2)
o2_98 = round(np.mean(o2)+2.06*np.std(o2),2)
o3_98 = round(np.mean(o3)+2.06*np.std(o3),2)
o4_98 = round(np.mean(o4)+2.06*np.std(o4),2)

plot_diagrama(o2_98, o1_98, o4_98, o3_98)

print("Content-Type:text/html")

utf8stdout = open(1, 'w', encoding='utf-8', closefd=False)

body = f"""
<body>

<div class="titulo negrita gris">Sobretensiones calculadas por la ANN</div>


<div id="main">

    <p class="negrita gris">CÁLCULOS DE LA RED:</p>
    <img src="utils/referencia.png?{random.random()}" alt="#" width=640 height=480 style="border: 3px solid #333; margin-top: 10px;"">

"""

print(body, file=utf8stdout)

footer = f"""

</div>


</body>

"""


print(footer, file=utf8stdout)