# ANN para cálculo de Sobretensiones de Frente Lento por Energización de Líneas

Para ocupar la aplicación:

1) Instalar Anaconda
2) Instalar PyCharm
3) Ocupar un intérprete de Python versión 3.7
4) Instalar paquetes numpy, tensorflow 2.0, scikit-learn, joblib, cgi, cgitb, matplotlib, random, pandas, sys (probablemente varios de estos paquetes ya vengan instalados por defecto y solo haya que instalar tensorflow 2.0 y scikit-learn)
5) Abrir una terminal en PyCharm (asegurándose de que el intérprete que se esté ocupando sea Python 3.7 con los paquetes mencionados anteriormente) y ejecutar el siguiente comando:   python -m http.server --bind localhost --cgi 8000 
6) Abrir su navegador favorito e ir a la siguiente dirección:    localhost:8000/

Si desea entrenar nuevamente la red (con más datos o datos nuevos):

1) Cierre la aplicación y dirigase a la carpeta del proyecto.
2) Reemplazar los archivos LIS de la carpeta simulaciones por las nuevas simulaciones que usted haya realizado. Deben tener el nombre "energizacion1.lis", "energizacion2.lis", etc.
3) Correr el código ann.py (ocupando el intérprete Python 3.7 con los paquetes ya mencionados)
4) Vuelva a abrir la aplicación según los pasos de la sección anterior. La aplicación leerá automáticamente la nueva red.
