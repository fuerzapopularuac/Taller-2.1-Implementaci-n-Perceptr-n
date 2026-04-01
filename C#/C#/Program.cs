using System;
using System.Collections.Generic;

class Perceptron
{
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private string activation;

    Random rand = new Random();

    public Perceptron(int inputSize, double lr = 0.1, int ep = 20, string act = "step")
    {
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
            weights[i] = rand.NextDouble() * 2 - 1;

        bias = rand.NextDouble() * 2 - 1;
        learningRate = lr;
        epochs = ep;
        activation = act;
    }

    private double Activate(double x)
    {
        switch (activation)
        {
            case "linear": return x;
            case "step": return x >= 0 ? 1 : 0;
            case "sigmoid": return 1 / (1 + Math.Exp(-x));
            case "relu": return Math.Max(0, x);
            case "tanh": return Math.Tanh(x);
            case "softmax": return Math.Exp(x) / (Math.Exp(x) + Math.Exp(-x));
            default: throw new Exception("Función inválida");
        }
    }

    public double Predict(double[] inputs)
    {
        double sum = bias;
        for (int i = 0; i < weights.Length; i++)
            sum += weights[i] * inputs[i];

        return Activate(sum);
    }

    public void Train(List<double[]> X, List<int> y)
    {
        for (int e = 0; e < epochs; e++)
        {
            Console.WriteLine($"\nEpoch {e + 1}");

            for (int i = 0; i < X.Count; i++)
            {
                double output = Predict(X[i]);

                if (activation != "step")
                    output = Math.Round(output);

                double error = y[i] - output;

                for (int j = 0; j < weights.Length; j++)
                    weights[j] += learningRate * error * X[i][j];

                bias += learningRate * error;

                Console.WriteLine($"Entrada: [{string.Join(",", X[i])}] → Esperado: {y[i]} → Predicho: {output}");
            }
        }

        Console.WriteLine("\nPesos finales: " + string.Join(", ", weights));
        Console.WriteLine("Bias final: " + bias);
    }
}

class Program
{
    static void Main()
    {
        var casos = new List<(string, List<double[]>, List<int>, string)>()
        {
            ("AND",
                new List<double[]> { new double[]{0,0}, new double[]{0,1}, new double[]{1,0}, new double[]{1,1} },
                new List<int>{0,0,0,1},
                "step"
            ),

            ("OR",
                new List<double[]> { new double[]{0,0}, new double[]{0,1}, new double[]{1,0}, new double[]{1,1} },
                new List<int>{0,1,1,1},
                "step"
            ),

            ("SPAM",
                new List<double[]> { new double[]{0,0}, new double[]{1,0}, new double[]{0,1}, new double[]{1,1} },
                new List<int>{0,1,1,1},
                "sigmoid"
            ),

            ("CLIMA",
                new List<double[]> { new double[]{0,0}, new double[]{0,1}, new double[]{1,0}, new double[]{1,1} },
                new List<int>{0,0,1,1},
                "tanh"
            ),

            ("FRAUDE",
                new List<double[]> { new double[]{0,0}, new double[]{1,0}, new double[]{0,1}, new double[]{1,1} },
                new List<int>{0,1,1,1},
                "relu"
            ),

            ("RIESGO ACADEMICO",
                new List<double[]>
                {
                    new double[]{10.0/20},
                    new double[]{12.0/20},
                    new double[]{13.0/20},
                    new double[]{14.0/20},
                    new double[]{16.0/20},
                    new double[]{18.0/20}
                },
                new List<int>{1,1,1,0,0,0},
                "step"
            )
        };

        foreach (var caso in casos)
        {
            Console.WriteLine("\n==============================");
            Console.WriteLine($"CASO: {caso.Item1} | Activación: {caso.Item4}");
            Console.WriteLine("==============================");

            var p = new Perceptron(caso.Item2[0].Length, 0.1, 20, caso.Item4);
            p.Train(caso.Item2, caso.Item3);

            Console.WriteLine("\nPruebas finales:");
            foreach (var x in caso.Item2)
            {
                double pred = p.Predict(x);
                if (caso.Item4 != "step")
                    pred = Math.Round(pred);

                Console.WriteLine($"[{string.Join(",", x)}] -> {pred}");
            }
        }
    }
}