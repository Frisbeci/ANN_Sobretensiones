function submit_inputs() {
    let long = document.getElementById('LONG').value;
    let rdivx = document.getElementById('RDIVX').value;
    let icc1f = document.getElementById('ICC1F').value;
    let icc3f = document.getElementById('ICC3F').value;
    let rinser = document.getElementById('RINSER').value;
    let rppos = document.getElementById('RPPOS').value;
    let rpcero = document.getElementById('RPCERO').value;
    let lppos = document.getElementById('LPPOS').value;
    let lpcero = document.getElementById('LPCERO').value;
    let cppos = document.getElementById('CPPOS').value;
    let cpcero = document.getElementById('CPCERO').value;
    let simet = document.getElementById('SIMET').value;
    let com1 = document.getElementById('COM1').value;
    let com2 = document.getElementById('COM2').value;

    let data = new FormData()
    data.append('long',long);
    data.append('rdivx',rdivx);
    data.append('icc1f',icc1f);
    data.append('icc3f',icc3f);
    data.append('rinser',rinser);
    data.append('rppos',rppos);
    data.append('rpcero',rpcero);
    data.append('lppos',lppos);
    data.append('lpcero',lpcero);
    data.append('cppos',cppos);
    data.append('cpcero',cpcero);
    data.append('simet',simet);
    data.append('com1',com1);
    data.append('com2',com2);

    let xhr = new XMLHttpRequest(); // AJAX
    xhr.open('POST','cgi-bin/app.py');

    xhr.onload = function(data) {
        window.alert('Sobretensiones calculadas exitosamente')
        let filas = data.currentTarget.responseText; //se recupera el texto que se imprime desde el .py que recupera los datos
        document.getElementById('resultados').innerHTML = filas; //se rellena la tabla del html
    }

    xhr.onerror = function () {
        alert('Error en el envío de datos')
    }

    console.log('Ajax enviado')

    xhr.send(data)
    return false; //para que el formulario nunca se envíe y la página no se recargue
}

function validate_form() {
    var simet = document.getElementById('SIMET').value;
    var com1 = document.getElementById('COM1').value;
    var com2 = document.getElementById('COM2').value;

    if (simet==0 && com1==1 && com2==1) {
        alert("Una compensación asimétrica no puede colocarse en ambos extremos de la línea");
        return false;
    }
    else {
        return submit_inputs();
    }
}