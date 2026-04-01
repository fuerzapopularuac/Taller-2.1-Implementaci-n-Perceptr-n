import java.util.function.Function;

public class Perceptron {
    private double[] pesos;
    private double bias;
    private double tasaAprendizaje;
    private Function<Double, Double> activacion;
    private String nombreActivacion;

    // Constructor
    public Perceptron(int numEntradas, double tasaAprendizaje, String tipoActivacion) {
        this.pesos = new double[numEntradas];
        this.bias = 0.0;
        this.tasaAprendizaje = tasaAprendizaje;
        
        // Inicializar pesos aleatorios entre -1 y 1
        for (int i = 0; i < numEntradas; i++) {
            this.pesos[i] = Math.random() * 2 - 1;
        }
        
        this.activacion = seleccionarActivacion(tipoActivacion);
        this.nombreActivacion = tipoActivacion;
    }
    
    // Seleccionar función de activación
    private Function<Double, Double> seleccionarActivacion(String tipo) {
        switch (tipo.toLowerCase()) {
            case "lineal":
                return x -> x;
            case "escalon":
                return x -> x >= 0 ? 1.0 : 0.0;
            case "sigmoidal":
                return x -> 1.0 / (1.0 + Math.exp(-x));
            case "relu":
                return x -> Math.max(0, x);
            case "softmax":
                // Para clasificación binaria, softmax se comporta como sigmoide
                return x -> 1.0 / (1.0 + Math.exp(-x));
            case "tanh":
                return x -> Math.tanh(x);
            default:
                return x -> x >= 0 ? 1.0 : 0.0;
        }
    }
    
    // Calcular sumatoria ponderada
    private double calcularSumatoria(double[] entradas) {
        double suma = bias;
        for (int i = 0; i < entradas.length; i++) {
            suma += pesos[i] * entradas[i];
        }
        return suma;
    }
    
    // Predecir salida para una entrada
    public double predecir(double[] entradas) {
        double sumatoria = calcularSumatoria(entradas);
        return activacion.apply(sumatoria);
    }
    
    // Entrenar el perceptrón
    public void entrenar(double[][] datos, double[] etiquetas, int iteraciones) {
        System.out.println("Entrenando con " + nombreActivacion + "...");
        
        for (int iter = 0; iter < iteraciones; iter++) {
            double errorTotal = 0;
            
            for (int i = 0; i < datos.length; i++) {
                double prediccion = predecir(datos[i]);
                double error = etiquetas[i] - prediccion;
                errorTotal += Math.abs(error);
                
                // Actualizar pesos y bias
                for (int j = 0; j < pesos.length; j++) {
                    pesos[j] += tasaAprendizaje * error * datos[i][j];
                }
                bias += tasaAprendizaje * error;
            }
            
            // Mostrar progreso cada 10 iteraciones
            if (iter % 10 == 0 || iter == iteraciones - 1) {
                System.out.printf("Iteración %d: Error total = %.4f%n", iter, errorTotal);
            }
        }
        System.out.println("Entrenamiento completado\n");
    }
    
    // Evaluar precisión
    public double evaluar(double[][] datos, double[] etiquetas) {
        int aciertos = 0;
        for (int i = 0; i < datos.length; i++) {
            double prediccion = predecir(datos[i]);
            // Redondear para comparar con etiquetas (por si es sigmoide)
            int clasePredicha = (prediccion >= 0.5) ? 1 : 0;
            if (clasePredicha == (int) etiquetas[i]) {
                aciertos++;
            }
        }
        return (double) aciertos / datos.length * 100;
    }
    
    // Mostrar pesos finales
    public void mostrarPesos() {
        System.out.print("Pesos finales: [");
        for (int i = 0; i < pesos.length; i++) {
            System.out.printf("%.4f", pesos[i]);
            if (i < pesos.length - 1) System.out.print(", ");
        }
        System.out.printf("], Bias: %.4f%n", bias);
    }
}