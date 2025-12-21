using MathNet.Numerics.LinearAlgebra;

namespace MLP
{
    public class NeuralNetwork
    {
        #region Private Variables
        private readonly int[] _layerSizes;
        private readonly int _numLayers;
        private readonly Random _rngRandom = new(42);
        
        #endregion

        #region parameters
        private readonly Matrix<double>[] _weights;
        private readonly Matrix<double>[] _biases;
        #endregion

        #region Caches
        private readonly Matrix<double>[] _A;
        private readonly Matrix<double>[] _Z;
        #endregion

        #region Ctor
        public NeuralNetwork(int[] layerSizes)
        {
            _layerSizes = layerSizes;
            _numLayers = layerSizes.Length;

            _weights = new Matrix<double>[_numLayers];
            _biases  = new Matrix<double>[_numLayers];
            _A = new Matrix<double>[_numLayers];
            _Z = new Matrix<double>[_numLayers];

            InitializeParameters();
        }

        #endregion

        #region Private Methods
        private void InitializeParameters()
        {
            for(var l = 1; l < _numLayers; l++) 
            {
                var std = Math.Sqrt(1.0 / _layerSizes[l-1]);
                var normal = new MathNet.Numerics.Distributions.Normal(0, std, _rngRandom);

                _weights[l] = Matrix<double>.Build
                .Random(_layerSizes[l], _layerSizes[l-1], normal);

                _biases[l] = Matrix<double>
                .Build
                .Dense(_layerSizes[l], 1);
            }
        }
        #endregion

        #region Public Methods
        public Matrix<double> ForwardPropagation(Matrix<double> X)
        {
            _A[0] = X;

            for (var l = 1; l < _numLayers; l++)
            {
                _Z[l] = Utils.AddBias(_weights[l] * _A[l - 1], _biases[l]);

                if (l < _numLayers - 1)  // Hidden layers: ReLU
                    _A[l] = Utils.ReLu(_Z[l]);
                else  // Output layer: Sigmoid
                    _A[l] = Utils.Sigmoid(_Z[l]);
            }

            return _A[_numLayers - 1]; // Y_hat
        }
        public (Matrix<double>[], Matrix<double>[]) BackPropagation(Matrix<double> Y)
        {
            var m = Y.ColumnCount;
            var dW = new Matrix<double>[_numLayers];
            var db = new Matrix<double>[_numLayers];
            
            var L = _numLayers - 1;
            
            var dZ = _A[L] - Y;  // Error(y_hat - y)
            
            for(var l = L; l >= 1; l--)
            {
                dW[l] = dZ * _A[l - 1].Transpose() / m;

                var z = dZ;
                
                db[l] = Matrix<double>.Build.Dense(
                    dZ.RowCount, 1,
                    (i, _) => z.Row(i).Sum() / m
                );
                
                if (l <= 1) continue;
                
                var dAprev = _weights[l].Transpose() * dZ;

                dZ = dAprev.PointwiseMultiply(Utils.ReLuDerivative(_Z[l - 1])); // Next layer error(dz[l-1])
            }
            
            return (dW, db);
        }
        public void UpdateParameters(Matrix<double>[] dW, Matrix<double>[] db, double learningRate)
        {
            for (var l = 1; l < _numLayers; l++)
            {
                _weights[l] -= learningRate * dW[l];
                _biases[l]  -= learningRate * db[l];
            }
        }
        public double ComputeAccuracy(Matrix<double> X, Matrix<double> Y)
        {
            var yHat = ForwardPropagation(X);
            
            var predictions = yHat.Map(y => y > 0.5 ? 1.0 : 0.0);
 
            var correct = 0.0;
            var m = Y.ColumnCount;
            for (var j = 0; j < m; j++)
            {
                if (Math.Abs(predictions[0, j] - Y[0, j]) < 0.001)
                    correct += 1;
            }
            return correct / m;
        }

        #endregion
    } 
}