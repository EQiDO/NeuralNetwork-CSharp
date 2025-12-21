using MathNet.Numerics.LinearAlgebra;

namespace MLP
{
    public static class Utils
    {
        public static Matrix<double> Sigmoid(Matrix<double> Z)
        {
            return Z.Map(z => 1.0 / (1.0 + Math.Exp(-z)));
        }
        
        public static Matrix<double> SigmoidDerivative(Matrix<double> A)
        {
            return A.PointwiseMultiply(A.Map(a => 1 - a));
        }
        public static Matrix<double> ReLu(Matrix<double> Z)
        {
            return Z.Map(z => Math.Max(0, z));
        }

        public static Matrix<double> ReLuDerivative(Matrix<double> Z)
        {
            return Z.Map(z => z > 0 ? 1.0 : 0.0);
        }
        public static Matrix<double> AddBias(Matrix<double> Z, Matrix<double> b)
        {
            var result = Z.Clone();
            for (var col = 0; col < Z.ColumnCount; col++)
            {
                for (var row = 0; row < Z.RowCount; row++)
                {
                    result[row, col] = Z[row, col] + b[row, 0];
                }
            }
            return result;
        }
        public static double LossFunction(Matrix<double> Y_hat, Matrix<double> Y)
        {
            var m = Y.ColumnCount;
            const double epsilon = 1e-8;

            var sum = 0.0;
            for (var j = 0; j < m; j++)
            {
                var y = Y[0, j];
                var yHat = Y_hat[0, j];

                sum += y * Math.Log(yHat + epsilon) + (1 - y) * Math.Log(1 - yHat + epsilon);
            }

            return -sum / m;
        }
    }
}