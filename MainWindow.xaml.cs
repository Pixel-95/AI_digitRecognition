using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Drawing;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using NumSharp;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using System.IO;
using System.Drawing;
using Tensorflow.Keras.Datasets;
using Microsoft.Win32;
using System.Windows.Interop;
using System.Diagnostics;

namespace TensorFlowNET_Keras
{
    public partial class MainWindow : Window
    {
        // Setup
        // 1. "<DisableWinExeOutputInference>true</DisableWinExeOutputInference>" in .csproj file to get Console application
        // 2. properties -> Make project Console Application
        // 3. Install NuGets:
        //      - TensorFlow.NET
        //      - SciSharp.TensorFlow.Redist
        //      - TensorFlow.Keras

        static string filepathNPZ = @"D:\Programme\MNISTdataset\mnist.npz";
        static string filepathModel = @"D:\Programme\TensorFlowNET-Keras\models\model";
        static string filepathFileDialog = @"D:\Dropbox\Downloads\";

        Tensorflow.Keras.Engine.Functional model;

        LayersApi layers = new LayersApi();

        static NDArray x_train, y_train, x_test, y_test;
        static Tensors inputs, outputs;

        public MainWindow()
        {
            InitializeComponent();

            //  ██╗
            //  ╚██╗ prepare dataset
            //  ██╔╝
            //  ╚═╝
            ((x_train, y_train), (x_test, y_test)) = load_data();
            //((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
            x_train = x_train / 255.0f;
            x_test = x_test / 255.0f;
            y_train = np_utils.to_categorical(y_train, 10);
            y_test = np_utils.to_categorical(y_test, 10);
            x_train = np.expand_dims(x_train, -1);
            x_test = np.expand_dims(x_test, -1);

            //  ██╗
            //  ╚██╗ declare NN layers
            //  ██╔╝
            //  ╚═╝
            // input layer
            inputs = keras.Input(shape: (28, 28, 1), name: "img");
            // convolutional layer
            var x = layers.Conv2D(32, 3, activation: "relu").Apply(inputs);
            x = layers.Conv2D(64, 3, activation: "relu").Apply(x);
            var block_1_output = layers.MaxPooling2D(3).Apply(x);
            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_1_output);
            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
            var block_2_output = layers.Add().Apply(new Tensors(x, block_1_output));
            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_2_output);
            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
            var block_3_output = layers.Add().Apply(new Tensors(x, block_2_output));
            x = layers.Conv2D(64, 3, activation: "relu").Apply(block_3_output);
            x = layers.GlobalAveragePooling2D().Apply(x);
            x = layers.Dense(256, activation: "relu").Apply(x);
            x = layers.Dropout(0.5f).Apply(x);
            // output layer
            outputs = layers.Dense(10).Apply(x);

            //  ██╗
            //  ╚██╗ create model
            //  ██╔╝
            //  ╚═╝
            // get model
            //model = CreateModel();
            //model.save_weights(filepathModel);

            // load model
            model = LoadModel();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            //  ██╗
            //  ╚██╗ evaluate model
            //  ██╔╝
            //  ╚═╝
            //Console.WriteLine();
            //model.evaluate(x_test, y_test);

            //  ██╗
            //  ╚██╗ predict single picture
            //  ██╔╝
            //  ╚═╝
            //predict
            int index = 1;
            Console.WriteLine();
            var predict = ((Tensor)model.predict(x_test[new Slice(index, index + 1)]))[0]; // 1D tensor

            // printing
            double sum = 0;
            double[] probabilities = new double[predict.shape[0]];
            for (int i = 0; i < predict.shape[0]; i++)
            {
                probabilities[i] = Math.Exp((float)predict[i]);
                sum += probabilities[i];
            }
            for (int i = 0; i < predict.shape[0]; i++)
            {
                probabilities[i] /= sum;
                Console.Write("P(" + i + ") = " + probabilities[i] * 100 + "%    \t<");
                for (int k = 1; k < probabilities[i] * 75; k++)
                    Console.Write("=");
                Console.WriteLine("");
            }

            Console.WriteLine("       ---------------------");
            Console.WriteLine("       " + probabilities.Sum() * 100 + "%");
            Console.WriteLine();

            for (int i = 0; i < predict.shape[0]; i++)
                if (y_test[index, i] == 1)
                    Console.WriteLine("correct answer: " + i);

            print("\ny = " + y_test[index]);
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            //  ██╗
            //  ╚██╗ file dialog
            //  ██╔╝
            //  ╚═╝
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "jpg Files (*.jpg)|*.jpg|All files (*.*)|*.*";
            openFileDialog.InitialDirectory = filepathFileDialog;
            if (openFileDialog.ShowDialog() != true)
                return;
            string file = openFileDialog.FileName;

            //  ██╗
            //  ╚██╗ make bitmap
            //  ██╔╝
            //  ╚═╝
            var uri = new Uri(file);
            var bitmapImage = new BitmapImage(uri);
            image.Source = bitmapImage;

            //  ██╗
            //  ╚██╗ crop bitmap to square and scale to 28x28
            //  ██╔╝
            //  ╚═╝
            CroppedBitmap cb = null;
            if (bitmapImage.PixelWidth < bitmapImage.PixelHeight)
                cb = new CroppedBitmap(bitmapImage, new Int32Rect(0, bitmapImage.PixelHeight / 2 - bitmapImage.PixelWidth / 2, bitmapImage.PixelWidth, bitmapImage.PixelWidth));
            else if (bitmapImage.PixelWidth > bitmapImage.PixelHeight)
                cb = new CroppedBitmap(bitmapImage, new Int32Rect(bitmapImage.PixelWidth / 2 - bitmapImage.PixelHeight / 2, 0, bitmapImage.PixelHeight, bitmapImage.PixelHeight));
            else
                cb = new CroppedBitmap(bitmapImage, new Int32Rect(0, 0, bitmapImage.PixelWidth, bitmapImage.PixelHeight));

            double scalingFactor = 28.0 / Math.Min(bitmapImage.PixelWidth, bitmapImage.PixelHeight);
            var targetBitmap = new TransformedBitmap(cb, new ScaleTransform(scalingFactor, scalingFactor));
            //image.Source = targetBitmap;

            //  ██╗
            //  ╚██╗ create float array from 0 to 3*255 (with filter, so only black and white -> no gray)
            //  ██╔╝
            //  ╚═╝
            // read bitmap
            float[,,] pictureArray = new float[1, 28, 28];
            for (int x = 0; x < 28; x++)
                for (int y = 0; y < 28; y++)
                {
                    var bytesPerPixel = (targetBitmap.Format.BitsPerPixel + 7) / 8;
                    var bytes = new byte[bytesPerPixel];
                    var rect = new Int32Rect(x, y, 1, 1);
                    targetBitmap.CopyPixels(rect, bytes, bytesPerPixel, 0);
                    float value = 3 * 255 - (bytes[2] + bytes[1] + bytes[0]);
                    pictureArray[0, y, x] = value;
                }

            // calc max and min and cut off gray
            float min = 3 * 255;
            float max = 0;
            for (int x = 0; x < 28; x++)
                for (int y = 0; y < 28; y++)
                {
                    if (pictureArray[0, y, x] > max)
                        max = pictureArray[0, y, x];
                    if (pictureArray[0, y, x] < min)
                        min = pictureArray[0, y, x];
                }
            float mean = (max + min) / 2;

            // make all pixels completly white or black
            for (int x = 0; x < 28; x++)
                for (int y = 0; y < 28; y++)
                {
                    if (pictureArray[0, y, x] > mean)
                        pictureArray[0, y, x] = 3 * 255.0f;
                    else
                        pictureArray[0, y, x] = 0;
                }

            //  ██╗
            //  ╚██╗ create numpy array
            //  ██╔╝
            //  ╚═╝
            NDArray NDimage = new NDArray(pictureArray);
            NDimage = NDimage / (3 * 255.0f);
            NDimage = np.expand_dims(NDimage, -1);
            Console.WriteLine();

            //  ██╗
            //  ╚██╗ predict
            //  ██╔╝
            //  ╚═╝
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            var predict = ((Tensor)model.predict(NDimage))[0]; // 1D tensor
            stopwatch.Stop();

            //  ██╗
            //  ╚██╗ printing
            //  ██╔╝
            //  ╚═╝
            double sum = 0;
            double[] probabilities = new double[predict.shape[0]];
            for (int i = 0; i < predict.shape[0]; i++)
            {
                probabilities[i] = Math.Exp((float)predict[i]);
                sum += probabilities[i];
            }
            int guess = -1;
            double bestProbability = -1;
            for (int i = 0; i < predict.shape[0]; i++)
            {
                if (probabilities[i] > bestProbability)
                {
                    bestProbability = probabilities[i];
                    guess = i;
                }

                probabilities[i] /= sum;
                Console.Write("P(" + i + ") = " + probabilities[i] * 100 + "%    \t<");
                for (int k = 1; k < probabilities[i] * 75; k++)
                    Console.Write("=");
                Console.WriteLine("");
            }

            Console.WriteLine("       ---------------------");
            Console.WriteLine("       " + probabilities.Sum() * 100 + "%");
            Console.WriteLine("\nThe digit is a " + guess + ".\n");
            digitText.Visibility = Visibility.Visible;

            switch (guess)
            {
                case 0:
                    number.Content = "zero";
                    break;
                case 1:
                    number.Content = "one";
                    break;
                case 2:
                    number.Content = "two";
                    break;
                case 3:
                    number.Content = "three";
                    break;
                case 4:
                    number.Content = "four";
                    break;
                case 5:
                    number.Content = "five";
                    break;
                case 6:
                    number.Content = "six";
                    break;
                case 7:
                    number.Content = "seven";
                    break;
                case 8:
                    number.Content = "eight";
                    break;
                case 9:
                    number.Content = "nine";
                    break;
            }

            certainty.Content = "Certainty = \n" + probabilities[guess] * 100 + "%\n\nCalculated in " + stopwatch.ElapsedMilliseconds + "ms";
        }

        // Support Stuff =======================================================================================================================================================================================
        static Tensorflow.Keras.Engine.Functional CreateModel()
        {
            // build keras model
            var model = keras.Model(inputs, outputs, name: "toy_resnet");
            model.summary();
            // compile keras model in tensorflow static graph
            model.compile(optimizer: keras.optimizers.Adam(1e-3f),
                loss: keras.losses.CategoricalCrossentropy(from_logits: true),
                metrics: new[] { "acc" });

            // training
            model.fit(x_train[new Slice(0, 60000)], y_train[new Slice(0, 60000)],
                      //x_train, y_train,
                      batch_size: 64,
                      epochs: 50,
                      validation_split: 0.2f);

            return model;
        }
        static Tensorflow.Keras.Engine.Functional LoadModel()
        {
            var model = keras.Model(inputs, outputs, name: "sequential");
            model.load_weights(filepathModel);
            model.compile(optimizer: keras.optimizers.Adam(1e-3f),
                loss: keras.losses.CategoricalCrossentropy(from_logits: true),
                metrics: new[] { "acc" });

            return model;
        }
        static DatasetPass load_data()
        {
            var bytes = File.ReadAllBytes(filepathNPZ);
            var datax = LoadX(bytes);
            var datay = LoadY(bytes);
            return new DatasetPass
            {
                Train = (datax.Item1, datay.Item1),
                Test = (datax.Item2, datay.Item2)
            };
        }
        static (NDArray, NDArray) LoadX(byte[] bytes)
        {
            var x = np.Load_Npz<byte[,,]>(bytes);
            return (x["x_train.npy"], x["x_test.npy"]);
        }
        static (NDArray, NDArray) LoadY(byte[] bytes)
        {
            var y = np.Load_Npz<byte[]>(bytes);
            return (y["y_train.npy"], y["y_test.npy"]);
        }
    }
}
