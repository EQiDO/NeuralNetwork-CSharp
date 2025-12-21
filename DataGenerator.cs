using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace MLP
{
    public class DataGenerator(int numTrain, int numTest)
    {
        public (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>)
        GenerateData()
        {
            var xTrain = CreateMatrix(3, numTrain);
            var yTrain = CalculateFunction(xTrain, numTrain);

            var xTest = CreateMatrix(3, numTest);
            var yTest = CalculateFunction(xTest, numTest);

            return (xTrain, yTrain, xTest, yTest);
        }


        private double BowlFunction(double x, double y)
        {
            return - Math.Pow(x - 3, 2)
            - Math.Pow(y - 5, 2)
            + 8;
        }

        private Matrix<double> CreateMatrix(int width, int height)
        {
            var uniform = new ContinuousUniform(-10, 20);
            return Matrix<double>.Build.Dense(width, height, (i, j) => uniform.Sample());
        }
        private Matrix<double> CalculateFunction(Matrix<double> xMatrix, int num)
        {
            var resultMatrix = Matrix<double>.Build.Dense(1, num);

            for (var j = 0; j < num; j++)
            {
                var x = xMatrix[0, j];
                var y = xMatrix[1, j];
                var z = xMatrix[2, j];

                var result = BowlFunction(x, y);
                resultMatrix[0, j] = z > result ? 1 : 0;
            }
            return resultMatrix;
        }
    }
}