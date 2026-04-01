class Perceptron {
    constructor(numEntradas, tasaAprendizaje, tipoActivacion) {
        this.pesos = [];
        this.bias = 0;
        this.tasaAprendizaje = tasaAprendizaje;
        this.tipoActivacion = tipoActivacion;
        
        // Inicializar pesos aleatorios entre -1 y 1
        for (let i = 0; i < numEntradas; i++) {
            this.pesos.push(Math.random() * 2 - 1);
        }
        
        // Asignar función de activación
        this.activacion = this.seleccionarActivacion(tipoActivacion);
    }
    
    // Seleccionar función de activación
    seleccionarActivacion(tipo) {
        switch (tipo.toLowerCase()) {
            case 'lineal':
                return (x) => x;
            case 'escalon':
                return (x) => x >= 0 ? 1 : 0;
            case 'sigmoidal':
                return (x) => 1 / (1 + Math.exp(-x));
            case 'relu':
                return (x) => Math.max(0, x);
            case 'softmax':
                // Para clasificación binaria, softmax se comporta como sigmoide
                return (x) => 1 / (1 + Math.exp(-x));
            case 'tanh':
                return (x) => Math.tanh(x);
            default:
                return (x) => x >= 0 ? 1 : 0;
        }
    }
    
    // Calcular sumatoria ponderada
    calcularSumatoria(entradas) {
        let suma = this.bias;
        for (let i = 0; i < entradas.length; i++) {
            suma += this.pesos[i] * entradas[i];
        }
        return suma;
    }
    
    // Predecir salida para una entrada
    predecir(entradas) {
        const sumatoria = this.calcularSumatoria(entradas);
        return this.activacion(sumatoria);
    }
    
    // Entrenar el perceptrón
    entrenar(datos, etiquetas, iteraciones) {
        console.log(`Entrenando con ${this.tipoActivacion}...`);
        
        for (let iter = 0; iter < iteraciones; iter++) {
            let errorTotal = 0;
            
            for (let i = 0; i < datos.length; i++) {
                const prediccion = this.predecir(datos[i]);
                const error = etiquetas[i] - prediccion;
                errorTotal += Math.abs(error);
                
                // Actualizar pesos y bias
                for (let j = 0; j < this.pesos.length; j++) {
                    this.pesos[j] += this.tasaAprendizaje * error * datos[i][j];
                }
                this.bias += this.tasaAprendizaje * error;
            }
            
            // Mostrar progreso cada 10 iteraciones
            if (iter % 10 === 0 || iter === iteraciones - 1) {
                console.log(`Iteración ${iter}: Error total = ${errorTotal.toFixed(4)}`);
            }
        }
        console.log('Entrenamiento completado\n');
    }
    
    // Evaluar precisión
    evaluar(datos, etiquetas) {
        let aciertos = 0;
        for (let i = 0; i < datos.length; i++) {
            const prediccion = this.predecir(datos[i]);
            // Redondear para comparar con etiquetas (por si es sigmoide)
            const clasePredicha = prediccion >= 0.5 ? 1 : 0;
            if (clasePredicha === etiquetas[i]) {
                aciertos++;
            }
        }
        return (aciertos / datos.length) * 100;
    }
    
    // Mostrar pesos finales
    mostrarPesos() {
        const pesosStr = this.pesos.map(p => p.toFixed(4)).join(', ');
        console.log(`Pesos finales: [${pesosStr}], Bias: ${this.bias.toFixed(4)}`);
    }
}

// Exportar para Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Perceptron;
}