public class Main {
    
    // Caso 1: AND lógico
    public static void testAND() {
        System.out.println("\n========== CASO 1: AND LÓGICO ==========");
        double[][] datos = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        double[] etiquetas = {0, 0, 0, 1};
        
        Perceptron p = new Perceptron(2, 0.1, "escalon");
        p.entrenar(datos, etiquetas, 20);
        p.mostrarPesos();
        
        System.out.println("\nPruebas:");
        for (double[] entrada : datos) {
            double salida = p.predecir(entrada);
            System.out.printf("%.0f AND %.0f = %.0f%n", entrada[0], entrada[1], salida);
        }
        System.out.printf("Precisión: %.2f%%%n", p.evaluar(datos, etiquetas));
    }
    
    // Caso 2: OR lógico
    public static void testOR() {
        System.out.println("\n========== CASO 2: OR LÓGICO ==========");
        double[][] datos = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        double[] etiquetas = {0, 1, 1, 1};
        
        Perceptron p = new Perceptron(2, 0.1, "escalon");
        p.entrenar(datos, etiquetas, 20);
        p.mostrarPesos();
        
        System.out.println("\nPruebas:");
        for (double[] entrada : datos) {
            double salida = p.predecir(entrada);
            System.out.printf("%.0f OR %.0f = %.0f%n", entrada[0], entrada[1], salida);
        }
        System.out.printf("Precisión: %.2f%%%n", p.evaluar(datos, etiquetas));
    }
    
    // Caso 3: Clasificación Spam (palabras_clave, longitud, enlaces)
    public static void testSpam() {
        System.out.println("\n========== CASO 3: SPAM vs NO SPAM ==========");
        double[][] datos = {
            {3, 150, 2},  // spam
            {0, 50, 0},   // no spam
            {2, 200, 1},  // spam
            {1, 80, 0},   // no spam
            {4, 300, 3},  // spam
            {0, 40, 0},   // no spam
            {5, 250, 4}   // spam
        };
        double[] etiquetas = {1, 0, 1, 0, 1, 0, 1};
        
        Perceptron p = new Perceptron(3, 0.05, "sigmoidal");
        p.entrenar(datos, etiquetas, 50);
        p.mostrarPesos();
        System.out.printf("Precisión: %.2f%%%n", p.evaluar(datos, etiquetas));
    }
    
    // Caso 4: Predicción del clima (temp, humedad, nubes)
    public static void testClima() {
        System.out.println("\n========== CASO 4: PREDICCIÓN DEL CLIMA (LLUVIA) ==========");
        double[][] datos = {
            {30, 80, 1},  // lluvia
            {25, 40, 0},  // no lluvia
            {28, 70, 1},  // lluvia
            {22, 50, 0},  // no lluvia
            {35, 90, 1},  // lluvia
            {20, 30, 0},  // no lluvia
            {32, 85, 1}   // lluvia
        };
        double[] etiquetas = {1, 0, 1, 0, 1, 0, 1};
        
        Perceptron p = new Perceptron(3, 0.05, "tanh");
        p.entrenar(datos, etiquetas, 50);
        p.mostrarPesos();
        System.out.printf("Precisión: %.2f%%%n", p.evaluar(datos, etiquetas));
    }
    
    // Caso 5: Detección de fraudes (monto, frecuencia, ubicación_riesgosa)
    public static void testFraudes() {
        System.out.println("\n========== CASO 5: DETECCIÓN DE FRAUDES ==========");
        double[][] datos = {
            {5000, 1, 1},   // fraude
            {50, 10, 0},    // no fraude
            {10000, 1, 1},  // fraude
            {100, 20, 0},   // no fraude
            {8000, 2, 1},   // fraude
            {200, 15, 0},   // no fraude
            {15000, 1, 1}   // fraude
        };
        double[] etiquetas = {1, 0, 1, 0, 1, 0, 1};
        
        Perceptron p = new Perceptron(3, 0.01, "sigmoidal");
        p.entrenar(datos, etiquetas, 50);
        p.mostrarPesos();
        System.out.printf("Precisión: %.2f%%%n", p.evaluar(datos, etiquetas));
    }
    
    // Caso 6: Riesgo académico (nota, asistencias, entregas)
    public static void testRiesgoAcademico() {
        System.out.println("\n========== CASO 6: RIESGO ACADÉMICO ==========");
        System.out.println("Nota: 0-20, Aprobado >= 14 | Riesgo: 1 = riesgo, 0 = sin riesgo");
        
        double[][] datos = {
            {18, 0.9, 0.8},  // sin riesgo
            {10, 0.4, 0.3},  // riesgo
            {12, 0.6, 0.5},  // riesgo
            {16, 0.8, 0.7},  // sin riesgo
            {8, 0.5, 0.4},   // riesgo
            {20, 1.0, 1.0},  // sin riesgo
            {5, 0.3, 0.2},   // riesgo
            {15, 0.7, 0.6}   // sin riesgo
        };
        double[] etiquetas = {0, 1, 1, 0, 1, 0, 1, 0};
        
        Perceptron p = new Perceptron(3, 0.05, "sigmoidal");
        p.entrenar(datos, etiquetas, 100);
        p.mostrarPesos();
        
        System.out.println("\nPruebas individuales:");
        System.out.println("Estudiante (nota=14, asistencias=0.7, entregas=0.6): " + 
                           (p.predecir(new double[]{14, 0.7, 0.6}) >= 0.5 ? "Riesgo" : "Sin riesgo"));
        System.out.println("Estudiante (nota=11, asistencias=0.5, entregas=0.4): " + 
                           (p.predecir(new double[]{11, 0.5, 0.4}) >= 0.5 ? "Riesgo" : "Sin riesgo"));
        
        System.out.printf("Precisión en entrenamiento: %.2f%%%n", p.evaluar(datos, etiquetas));
    }
    
    // Probar con diferentes funciones de activación para un caso
    public static void testComparacionActivaciones() {
        System.out.println("\n========== COMPARACIÓN DE FUNCIONES DE ACTIVACIÓN ==========");
        System.out.println("Caso: AND Lógico");
        
        double[][] datosAND = {{0,0}, {0,1}, {1,0}, {1,1}};
        double[] etiquetasAND = {0, 0, 0, 1};
        
        String[] activaciones = {"lineal", "escalon", "sigmoidal", "relu", "softmax", "tanh"};
        
        for (String act : activaciones) {
            Perceptron p = new Perceptron(2, 0.1, act);
            p.entrenar(datosAND, etiquetasAND, 30);
            double precision = p.evaluar(datosAND, etiquetasAND);
            System.out.printf("%-10s -> Precisión: %.2f%%%n", act, precision);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("=== IMPLEMENTACIÓN DE PERCEPTRÓN EN JAVA ===");
        System.out.println("Integrantes: [Tu nombre] y [Mi nombre]");
        
        // Ejecutar todos los casos
        testAND();
        testOR();
        testSpam();
        testClima();
        testFraudes();
        testRiesgoAcademico();
        testComparacionActivaciones();
        
        System.out.println("\n=== FIN DE LAS PRUEBAS ===");
    }
}