const Perceptron = require('./perceptron.js');

// Caso 1: AND lógico
function testAND() {
    console.log('\n========== CASO 1: AND LÓGICO ==========');
    const datos = [
        [0, 0], [0, 1], [1, 0], [1, 1]
    ];
    const etiquetas = [0, 0, 0, 1];
    
    const p = new Perceptron(2, 0.1, 'escalon');
    p.entrenar(datos, etiquetas, 20);
    p.mostrarPesos();
    
    console.log('\nPruebas:');
    datos.forEach(entrada => {
        const salida = p.predecir(entrada);
        console.log(`${entrada[0]} AND ${entrada[1]} = ${Math.round(salida)}`);
    });
    console.log(`Precisión: ${p.evaluar(datos, etiquetas).toFixed(2)}%`);
}

// Caso 2: OR lógico
function testOR() {
    console.log('\n========== CASO 2: OR LÓGICO ==========');
    const datos = [
        [0, 0], [0, 1], [1, 0], [1, 1]
    ];
    const etiquetas = [0, 1, 1, 1];
    
    const p = new Perceptron(2, 0.1, 'escalon');
    p.entrenar(datos, etiquetas, 20);
    p.mostrarPesos();
    
    console.log('\nPruebas:');
    datos.forEach(entrada => {
        const salida = p.predecir(entrada);
        console.log(`${entrada[0]} OR ${entrada[1]} = ${Math.round(salida)}`);
    });
    console.log(`Precisión: ${p.evaluar(datos, etiquetas).toFixed(2)}%`);
}

// Caso 3: Clasificación Spam (palabras_clave, longitud, enlaces)
function testSpam() {
    console.log('\n========== CASO 3: SPAM vs NO SPAM ==========');
    const datos = [
        [3, 150, 2],  // spam
        [0, 50, 0],   // no spam
        [2, 200, 1],  // spam
        [1, 80, 0],   // no spam
        [4, 300, 3],  // spam
        [0, 40, 0],   // no spam
        [5, 250, 4]   // spam
    ];
    const etiquetas = [1, 0, 1, 0, 1, 0, 1];
    
    const p = new Perceptron(3, 0.05, 'sigmoidal');
    p.entrenar(datos, etiquetas, 50);
    p.mostrarPesos();
    console.log(`Precisión: ${p.evaluar(datos, etiquetas).toFixed(2)}%`);
}

// Caso 4: Predicción del clima (temp, humedad, nubes)
function testClima() {
    console.log('\n========== CASO 4: PREDICCIÓN DEL CLIMA (LLUVIA) ==========');
    const datos = [
        [30, 80, 1],  // lluvia
        [25, 40, 0],  // no lluvia
        [28, 70, 1],  // lluvia
        [22, 50, 0],  // no lluvia
        [35, 90, 1],  // lluvia
        [20, 30, 0],  // no lluvia
        [32, 85, 1]   // lluvia
    ];
    const etiquetas = [1, 0, 1, 0, 1, 0, 1];
    
    const p = new Perceptron(3, 0.05, 'tanh');
    p.entrenar(datos, etiquetas, 50);
    p.mostrarPesos();
    console.log(`Precisión: ${p.evaluar(datos, etiquetas).toFixed(2)}%`);
}

// Caso 5: Detección de fraudes (monto, frecuencia, ubicación_riesgosa)
function testFraudes() {
    console.log('\n========== CASO 5: DETECCIÓN DE FRAUDES ==========');
    const datos = [
        [5000, 1, 1],   // fraude
        [50, 10, 0],    // no fraude
        [10000, 1, 1],  // fraude
        [100, 20, 0],   // no fraude
        [8000, 2, 1],   // fraude
        [200, 15, 0],   // no fraude
        [15000, 1, 1]   // fraude
    ];
    const etiquetas = [1, 0, 1, 0, 1, 0, 1];
    
    const p = new Perceptron(3, 0.01, 'sigmoidal');
    p.entrenar(datos, etiquetas, 50);
    p.mostrarPesos();
    console.log(`Precisión: ${p.evaluar(datos, etiquetas).toFixed(2)}%`);
}

// Caso 6: Riesgo académico (nota, asistencias, entregas)
function testRiesgoAcademico() {
    console.log('\n========== CASO 6: RIESGO ACADÉMICO ==========');
    console.log('Nota: 0-20, Aprobado >= 14 | Riesgo: 1 = riesgo, 0 = sin riesgo');
    
    const datos = [
        [18, 0.9, 0.8],  // sin riesgo
        [10, 0.4, 0.3],  // riesgo
        [12, 0.6, 0.5],  // riesgo
        [16, 0.8, 0.7],  // sin riesgo
        [8, 0.5, 0.4],   // riesgo
        [20, 1.0, 1.0],  // sin riesgo
        [5, 0.3, 0.2],   // riesgo
        [15, 0.7, 0.6]   // sin riesgo
    ];
    const etiquetas = [0, 1, 1, 0, 1, 0, 1, 0];
    
    const p = new Perceptron(3, 0.05, 'sigmoidal');
    p.entrenar(datos, etiquetas, 100);
    p.mostrarPesos();
    
    console.log('\nPruebas individuales:');
    const caso1 = p.predecir([14, 0.7, 0.6]) >= 0.5 ? 'Riesgo' : 'Sin riesgo';
    console.log(`Estudiante (nota=14, asistencias=0.7, entregas=0.6): ${caso1}`);
    
    const caso2 = p.predecir([11, 0.5, 0.4]) >= 0.5 ? 'Riesgo' : 'Sin riesgo';
    console.log(`Estudiante (nota=11, asistencias=0.5, entregas=0.4): ${caso2}`);
    
    console.log(`Precisión en entrenamiento: ${p.evaluar(datos, etiquetas).toFixed(2)}%`);
}

// Probar con diferentes funciones de activación para un caso
function testComparacionActivaciones() {
    console.log('\n========== COMPARACIÓN DE FUNCIONES DE ACTIVACIÓN ==========');
    console.log('Caso: AND Lógico');
    
    const datosAND = [[0,0], [0,1], [1,0], [1,1]];
    const etiquetasAND = [0, 0, 0, 1];
    
    const activaciones = ['lineal', 'escalon', 'sigmoidal', 'relu', 'softmax', 'tanh'];
    
    activaciones.forEach(act => {
        const p = new Perceptron(2, 0.1, act);
        p.entrenar(datosAND, etiquetasAND, 30);
        const precision = p.evaluar(datosAND, etiquetasAND);
        console.log(`${act.padEnd(10)} -> Precisión: ${precision.toFixed(2)}%`);
    });
}

// Función principal
function main() {
    console.log('=== IMPLEMENTACIÓN DE PERCEPTRÓN EN JAVASCRIPT ===');
    console.log('Integrantes: [Tu nombre] y [Mi nombre]');
    
    // Ejecutar todos los casos
    testAND();
    testOR();
    testSpam();
    testClima();
    testFraudes();
    testRiesgoAcademico();
    testComparacionActivaciones();
    
    console.log('\n=== FIN DE LAS PRUEBAS ===');
}

// Ejecutar
main();