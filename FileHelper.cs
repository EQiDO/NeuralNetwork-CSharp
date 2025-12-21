using System.Diagnostics;
using System.Globalization;
using MathNet.Numerics.LinearAlgebra;

namespace MLP
{
    public static class FileHelper
    {
        public static void DeleteCsvFiles()
        {
            string[] files = ["x_train.csv", "y_train.csv", "x_test.csv", "y_test.csv"];
        
            foreach (var file in files)
            {
                if (!File.Exists(file))
                    continue;
                File.Delete(file);
                Console.WriteLine($"Deleted: {file}");
            }
        }
    
        public static void SaveToCsv(this Matrix<double> matrix, string filePath)
        {
            using var writer = new StreamWriter(filePath);
            for (int i = 0; i < matrix.RowCount; i++)
            {
                var row = new List<string>();
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    row.Add(matrix[i, j].ToString("G17", CultureInfo.InvariantCulture));
                }
                writer.WriteLine(string.Join(",", row));
            }
        }
        public static void RunPythonVisualization(string filename)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "python",  
                        Arguments = filename,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };
            
                process.Start();
            
                string output = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();
            
                process.WaitForExit();
            
                if (process.ExitCode != 0)
                {
                    Console.WriteLine($"Python Error: {error}");
                }
                else
                {
                    Console.WriteLine("Visualization completed successfully.");
                    if (!string.IsNullOrEmpty(output))
                        Console.WriteLine(output);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to run Python: {ex.Message}");
                Console.WriteLine("Make sure Python is installed and in PATH.");
            }
        }
    }
}