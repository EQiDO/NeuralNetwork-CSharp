using System.Diagnostics;
using System.Globalization;
using MathNet.Numerics.LinearAlgebra;

namespace MLP
{
    static class Program
    {
        private const int Epochs = 100;
        private const double LearningRate = 0.01;
        private const double Tolerance = 1e-3;

        private static StreamWriter? _logWriter;
        
        private static void Main()
        {
            var (xTrain, yTrain, xTest, yTest) = CreateDataset();

            PrintDatasetStats("Train", yTrain);
            PrintDatasetStats("Test", yTest);

            var network = new NeuralNetwork([3, 10, 10, 10, 1]);

            Train(network, xTrain, yTrain, xTest, yTest);
            // PlotDataSet(xTrain, yTrain, xTest, yTest);

            PrintFinalResults(network, xTrain, yTrain, xTest, yTest);
            
            AppDomain.CurrentDomain.ProcessExit += RaiseEventOnProcessExit;
        }

        private static (Matrix<double> xTrain, Matrix<double> yTrain,
                        Matrix<double> xTest, Matrix<double> yTest)
            CreateDataset()
        {
            var generator = new DataGenerator(10_000, 2_000);
            return generator.GenerateData();
        }

        private static void PrintDatasetStats(string name, Matrix<double> y)
        {
            var ones = CountOnes(y);
            var total = y.ColumnCount;

            Console.WriteLine(
                $"{name} dataset: {ones} ones ({(double)ones / total:P2}), {total - ones} zeros");
        }
        private static void PlotDataSet(Matrix<double> xTrain, Matrix<double> yTrain,
            Matrix<double> xTest, Matrix<double> yTest)
        {
            xTrain.SaveToCsv("x_train.csv");
            yTrain.SaveToCsv("y_train.csv");
            xTest.SaveToCsv("x_test.csv");
            yTest.SaveToCsv("y_test.csv");
            FileHelper.RunPythonVisualization("DataVisualization.py");
        }
        private static int CountOnes(Matrix<double> y)
        {
            var count = 0;
            for (var i = 0; i < y.ColumnCount; i++)
            {
                if (Math.Abs(y[0, i] - 1) < Tolerance)
                    count++;
            }
            return count;
        }
        
        private static void Train(
            NeuralNetwork nn,
            Matrix<double> xTrain,
            Matrix<double> yTrain,
            Matrix<double> xTest,
            Matrix<double> yTest)
        {
            Console.WriteLine("\nStarting training...\n");

            _logWriter = new StreamWriter("training_log.csv");
            _logWriter.WriteLine("epoch,loss,train_accuracy");

            var timer = Stopwatch.StartNew();

            for (var epoch = 1; epoch <= Epochs; epoch++)
            {
                var yHat = nn.ForwardPropagation(xTrain);
                var loss = Utils.LossFunction(yHat, yTrain);

                var (dW, db) = nn.BackPropagation(yTrain);
                nn.UpdateParameters(dW, db, LearningRate);

                var trainAcc = nn.ComputeAccuracy(xTrain, yTrain);

                _logWriter.WriteLine(
                    $"{epoch}," +
                    $"{loss.ToString(CultureInfo.InvariantCulture)}," +
                    $"{trainAcc.ToString(CultureInfo.InvariantCulture)}"
                );

                if (epoch == 1 || epoch % 10 == 0)
                    LogEpoch(epoch, loss, nn, xTrain, yTrain, xTest, yTest);
            }

            timer.Stop();

            _logWriter.Flush();
            _logWriter.Close();

            Console.WriteLine("\nTraining completed!");
            Console.WriteLine($"Run time: {timer.ElapsedMilliseconds} ms\n");
            FileHelper.RunPythonVisualization("PlotData.py");
        }

        private static void LogEpoch(
            int epoch,
            double loss,
            NeuralNetwork nn,
            Matrix<double> xTrain,
            Matrix<double> yTrain,
            Matrix<double> xTest,
            Matrix<double> yTest)
        {
            var trainAcc = nn.ComputeAccuracy(xTrain, yTrain);
            var testAcc = nn.ComputeAccuracy(xTest, yTest);

            Console.WriteLine(
                $"Epoch {epoch,4}: Loss={loss:F6}, TrainAcc={trainAcc:F4}, TestAcc={testAcc:F4}");
        }

        private static void PrintFinalResults(
            NeuralNetwork nn,
            Matrix<double> xTrain,
            Matrix<double> yTrain,
            Matrix<double> xTest,
            Matrix<double> yTest)
        {
            Console.WriteLine("Final Results:");
            Console.WriteLine($"Train Accuracy: {nn.ComputeAccuracy(xTrain, yTrain):F4}");
            Console.WriteLine($"Test Accuracy:  {nn.ComputeAccuracy(xTest, yTest):F4}");
        }

        private static void RaiseEventOnProcessExit(object? sender, EventArgs e)
        {
            FileHelper.DeleteCsvFiles();
        }
    }
}
