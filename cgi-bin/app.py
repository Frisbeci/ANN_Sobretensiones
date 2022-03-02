# !/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np # Python 3.7, tensorflow 2.0, scikit-learn; importar automaticamente from tensorflow.keras.models import load_model y from joblib import load ;;;;;;;;  python -m http.server --bind localhost --cgi 8000
from tensorflow.keras.models import load_model
from joblib import load
import cgi
import cgitb
import matplotlib.pyplot as plt
import random

def plot_SOV(ft_REC, ff_REC,v_mag=500,frec=50,tmax=0.2):
    w = 2*np.pi*frec
    t = np.linspace(0, tmax, 10000)
    step = np.heaviside(t-0.05,0.5)
    exp_ft = np.clip(np.exp(-(t-0.05-1/50*1/12)*200)*(ft_REC-1)+1,0,ft_REC)
    exp_ff = np.clip(np.exp(-(t-0.05-1/50*1/12)*200)*(ff_REC-1)+1,0,ff_REC)
    va = v_mag*np.sqrt(2/3)*np.sin(w*t)*step*exp_ft
    vb = v_mag*np.sqrt(2/3)*np.sin(w*t-np.pi*2/3)*step*exp_ft
    vc = v_mag*np.sqrt(2/3)*np.sin(w*t+np.pi*2/3)*step*exp_ft
    vab = v_mag*np.sqrt(2)*np.sin(w*t)*step*exp_ff
    vbc = v_mag*np.sqrt(2)*np.sin(w*t-np.pi*2/3)*step*exp_ff
    vca = v_mag*np.sqrt(2)*np.sin(w*t+np.pi*2/3)*step*exp_ff
    plt.subplot(2,1,1)
    plt.title('Energización | Referencia sobretensiones lado receptor')
    plt.grid(alpha = 0.5, linestyle = '--')
    plt.plot(t*1000,va, color= 'tab:red')
    plt.plot(t*1000,vb, color= 'tab:green')
    plt.plot(t*1000,vc, color= 'tab:blue')
    plt.ylabel('Receptor f-n [kV]')
    plt.xlabel('Tiempo [ms]')
    plt.xlim([0,tmax*1e3])
    plt.subplot(2,1,2)
    plt.grid(alpha = 0.5, linestyle = '--')
    plt.plot(t*1000,vab, color= 'tab:red')
    plt.plot(t*1000,vbc, color= 'tab:green')
    plt.plot(t*1000,vca, color= 'tab:blue')
    plt.ylabel('Receptor f-f [kV]')
    plt.xlabel('Tiempo [ms]')
    plt.xlim([0,tmax*1e3])
    plt.tight_layout()
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
    o1 = np.append(o1,round(float(calculos['Predicciones'][0])/np.sqrt(3), 4))
    o2 = np.append(o2,round(float(calculos['Predicciones'][1])/np.sqrt(3), 4))
    o3 = np.append(o3,round(float(calculos['Predicciones'][2])/np.sqrt(3), 4))
    o4 = np.append(o4,round(float(calculos['Predicciones'][3])/np.sqrt(3), 4))


o1_98 = round(np.mean(o1)+2.06*np.std(o1),4)
o2_98 = round(np.mean(o2)+2.06*np.std(o2),4)
o3_98 = round(np.mean(o3)+2.06*np.std(o3),4)
o4_98 = round(np.mean(o4)+2.06*np.std(o4),4)

plot_SOV(o3_98, o4_98)

print("Content-Type:text/html")

utf8stdout = open(1, 'w', encoding='utf-8', closefd=False)

body = f"""
<body>

<div class="titulo negrita gris">Sobretensiones calculadas por la ANN</div>


<div id="main">

<div class="flex-container">
    <div class="flex-child magenta">
    <p class="negrita gris">IMAGEN REFERENCIAL:</p>
    <img src="utils/referencia.png?{random.random()}" alt="#" width=500 height=350 style="border: 3px solid #333; margin-top: 10px;"">
    </div>
"""

print(body, file=utf8stdout)

print(f'''
<div class="flex-child green">
''', file=utf8stdout)

print(f'''<p class='negrita gris'> CÁLCULOS DE LA RED: </p>''', file=utf8stdout)

print(f'''<p class='negrita'> Fase-Neutro Emisor [pu]: </p> <p>  {o1_98} </p>''', file=utf8stdout)
print(f'''<p class='negrita'> Fase-Fase Emisor [pu]: </p> <p>  {o2_98} </p>''', file=utf8stdout)
print(f'''<p class='negrita'> Fase-Neutro Receptor [pu]: </p> <p>  {o3_98} </p>''', file=utf8stdout)
print(f'''<p class='negrita'> Fase-Fase Receptor [pu]: </p> <p>  {o4_98} </p>''', file=utf8stdout)

print(f'''
</div>
</div>
''',  file=utf8stdout)

footer = f"""

</div>


</body>

"""


print(footer, file=utf8stdout)