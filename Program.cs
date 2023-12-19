
namespace NeuralNetwork
{

    internal class Program
    {
        /*

        /////////ACCURACY AFTER 5 EPOCHS ~~ 81%/////////

        */
        public static string mnistDataPath = "data/";
        static void LoadMNIST(string path, out double[][] inputs, out double[] outputs)
        {
            inputs = new double[60000][];
            outputs = new double[60000];

            using (FileStream inputfsImages = new FileStream(Path.Combine(path, "train-images-idx3-ubyte"), FileMode.Open))
            {
                using (BinaryReader brImages = new BinaryReader(inputfsImages))
                {
                    brImages.ReadBytes(16); //skip header
                    for (int i = 0; i < 60000; i++)
                    {
                        inputs[i] = new double[784];
                        for (int j = 0; j < 784; j++)
                        {
                            inputs[i][j] = brImages.ReadByte() / 255.0;
                        }
                    }
                }
            }

            using (FileStream inputfsLabels = new FileStream(Path.Combine(path, "train-labels-idx1-ubyte"), FileMode.Open))
            {
                using (BinaryReader brLabels = new BinaryReader(inputfsLabels))
                {
                    brLabels.ReadBytes(8); //skip header
                    for (int i = 0; i < 60000; i++)
                    {
                        outputs[i] = brLabels.ReadByte();
                    }
                }
            }

        }
        private static double[,] InitializeWeights(int numRows, int numCols)
        {
            double[,] weights = new double[numRows, numCols];
            Random random = new Random();

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    weights[i, j] = random.NextDouble() - 0.5;
                }
            }

            return weights;
        }
        #region MatrixOperations


        private static double[,] MatrixAdd(double[,] matrix1, double[,] matrix2)
        {
            int rows = matrix1.GetLength(0);
            int cols = matrix1.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix1[i, j] + matrix2[i, j];
                }
            }

            return result;
        }

        private static double[,] MatrixSubtract(double[,] matrix1, double[,] matrix2)
        {
            int rows = matrix1.GetLength(0);
            int cols = matrix1.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix1[i, j] - matrix2[i, j];
                }
            }

            return result;
        }

        private static double[,] MatrixMultiply(double[,] matrix, double scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] * scalar;
                }
            }

            return result;
        }

        private static double[,] MatrixMultiply(double scalar, double[,] matrix)
        {
            return MatrixMultiply(matrix, scalar);
        }

        private static double[,] MatrixMultiply(double[,] matrix1, double[,] matrix2)
        {
            int rows1 = matrix1.GetLength(0);
            int cols1 = matrix1.GetLength(1);
            int cols2 = matrix2.GetLength(1);
            double[,] result = new double[rows1, cols2];

            for (int i = 0; i < rows1; i++)
            {
                for (int j = 0; j < cols2; j++)
                {
                    double sum = 0;

                    for (int k = 0; k < cols1; k++)
                    {
                        sum += matrix1[i, k] * matrix2[k, j];
                    }

                    result[i, j] = sum;
                }
            }

            return result;
        }

        private static double[,] MatrixTranspose(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[,] result = new double[cols, rows];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }

            return result;
        }

        private static double[,] ElementwiseMultiply(double[,] matrix1, double[,] matrix2)
        {
            int rows = matrix1.GetLength(0);
            int cols = matrix1.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix1[i, j] * matrix2[i, j];
                }
            }

            return result;
        }

        private static double[,] ElementwiseMultiply(double[,] matrix, double scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] * scalar;
                }
            }

            return result;
        }
        private static int ArgMax(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            int maxIndex = 0;
            double maxVal = matrix[0, 0];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (matrix[i, j] > maxVal)
                    {
                        maxVal = matrix[i, j];
                        maxIndex = i;
                    }
                }
            }

            return maxIndex;
        }
        private static double[,] MatrixAddColumn(double[] vector)
        {
            int rows = vector.Length;
            double[,] matrix = new double[rows, 1];

            for (int i = 0; i < rows; i++)
            {
                matrix[i, 0] = vector[i];
            }

            return matrix;
        }
        private static double[,] ElementwiseSubtract(double scalar, double[,] matrix1)
        {
            int rows = matrix1.GetLength(0);
            int cols = matrix1.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalar - matrix1[i, j];
                }
            }

            return result;
        }
        #endregion
        private static double[,] Sigmoid(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = 1.0 / (1.0 + Math.Exp(-matrix[i, j]));
                }
            }

            return result;
        }

        private static double[,] CalculateError(double[,] output, double label)
        {
            int rows = output.GetLength(0);
            int cols = output.GetLength(1);
            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = 1.0 / rows * Math.Pow(output[i, j] - label, 2);
                }
            }

            return result;
        }

        static void Main(string[] args)
        {
            double[][] inputs;
            double[] outputs;
            LoadMNIST(mnistDataPath, out inputs, out outputs);

            int numHiddenNodes = 20;
            int numOutputNodes = 10;

            double[,] wInputHiddenLayer = InitializeWeights(numHiddenNodes, 784);
            double[,] wHiddenLayerOutput = InitializeWeights(numOutputNodes, numHiddenNodes);
            double[,] biasInputHiddenLayer = new double[numHiddenNodes, 1];
            double[,] biasHiddenLayerOutput = new double[numOutputNodes, 1];

            double learnRate = 0.01;
            int numCorrect = 0;
            int epochs = 10;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    double[] img = inputs[i];
                    double labelsCurrent = outputs[i];

                    double[,] imgMatrix = MatrixAddColumn(img);
                    labelsCurrent += 1;

                    double[,] hiddenLayerPre = MatrixAdd(MatrixMultiply(wInputHiddenLayer, imgMatrix), biasInputHiddenLayer);
                    double[,] hiddenLayer = Sigmoid(hiddenLayerPre);

                    double[,] outputsCurrentPre = MatrixAdd(MatrixMultiply(wHiddenLayerOutput, hiddenLayer), biasHiddenLayerOutput);
                    double[,] outputsCurrent = Sigmoid(outputsCurrentPre);

                    double[,] mse = CalculateError(outputsCurrent, labelsCurrent);
                    numCorrect += (ArgMax(outputsCurrent) == (int)labelsCurrent) ? 1 : 0;

                    double[] deltaOutputs = new double[numOutputNodes];
                    for (int j = 0; j < numOutputNodes; j++)
                    {
                        deltaOutputs[j] = 1.0 / numOutputNodes * (outputsCurrent[j, 0] - (j == (int)labelsCurrent ? 1.0 : 0.0));
                    }

                    wHiddenLayerOutput = MatrixAdd(wHiddenLayerOutput, MatrixMultiply(-learnRate, MatrixMultiply(MatrixAddColumn(deltaOutputs), MatrixTranspose(hiddenLayer))));
                    biasHiddenLayerOutput = MatrixAdd(biasHiddenLayerOutput, MatrixMultiply(-learnRate, MatrixAddColumn(deltaOutputs)));

                    double[,] deltaHiddenLayer = ElementwiseMultiply(MatrixMultiply(MatrixTranspose(wHiddenLayerOutput), MatrixAddColumn(deltaOutputs)), ElementwiseMultiply(hiddenLayer, ElementwiseSubtract(1, hiddenLayer)));
                    wInputHiddenLayer = MatrixAdd(wInputHiddenLayer, MatrixMultiply(-learnRate, MatrixMultiply(deltaHiddenLayer, MatrixTranspose(imgMatrix))));
                    biasInputHiddenLayer = MatrixAdd(biasInputHiddenLayer, MatrixMultiply(-learnRate, deltaHiddenLayer));


                }

                Console.WriteLine($"Accuracy: {Math.Round((double)numCorrect / inputs.Length * 100, 2)}%");
                numCorrect = 0;
            }
        }
    }
}