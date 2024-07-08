using NumSharp;
using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace SharpKAN_lib
{
    internal class Linear_Regression : nn.Module<Tensor, Tensor>
    {
        private Linear linear;
        public Linear_Regression() : base(nameof(Linear_Regression))
        {
            linear = torch.nn.Linear(1, 1, dtype: float64);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return linear.forward(x);
        }

        public double[] get_params()
        {
            return new double[] { linear.weight.ToDouble(), linear.bias.ToDouble() };
        }
    }
    public class Utils
    {
        public static Dictionary<string, Func<Tensor, Tensor>> SYMBOLIC_LIB = new Dictionary<string, Func<Tensor, Tensor>>()
        {
                {"x", (Tensor x) => x},
                {"x^2", (Tensor x) => torch.square(x)},
                {"x^3", (Tensor x) => torch.pow(x, 3)},
                {"x^4", (Tensor x) => torch.pow(x, 4)},
                {"1/x", (Tensor x) => 1/ x},
                {"1/x^2", (Tensor x) => 1 / torch.square(x)},
                {"1/x^3", (Tensor x) => 1 / torch.pow(x, 3)},
                {"1/x^4", (Tensor x) => 1 / torch.pow(x, 4)},
                {"sqrt", (Tensor x) => torch.sqrt(x)},
                {"1/sqrt(x)", (Tensor x) => 1 / torch.sqrt(x)},
                {"exp", (Tensor x) => torch.exp(x)},
                {"log", (Tensor x) => torch.log(x)},
                {"abs", (Tensor x) => torch.abs(x)},
                {"sin", (Tensor x) => torch.sin(x)},
                //{"cos", (Tensor x) => torch.cos(x)},
                {"tan", (Tensor x) => torch.tan(x)},
                {"sinh", (Tensor x) => torch.sinh(x)},
                {"cosh", (Tensor x) => torch.cosh(x)},
                {"tanh", (Tensor x) => torch.tanh(x)},
                {"sigmoid", (Tensor x) => torch.sigmoid(x)},
                {"sgn", (Tensor x) => torch.sign(x)},
                {"arcsin", (Tensor x) => torch.arcsin(x)},
                {"arccos", (Tensor x) => torch.arccos(x)},
                {"arctan", (Tensor x) => torch.arctan(x)},
                {"arctanh", (Tensor x) => torch.arctanh(x)},
                {"0", (Tensor x) => x*0},
                {"gaussian", (Tensor x) => torch.exp(torch.square(-x))},
        };

        public static Dictionary<string, Tensor> create_dataset(Func<Tensor, Tensor> f, NDArray ranges = null, int n_var = 2, int train_num = 1000, int test_num = 1000,
                                bool normalize_input = false, bool normalize_label = false, string device = "cpu", int seed = 0)
        {
            np.random.seed(seed);
            torch.manual_seed(seed);

            torch.set_default_dtype(torch.float64);

            if (ranges == null)
            {
                ranges = new NDArray(values: new double[] { -1.0, 1.0 });
            }

            if (ranges.shape.Length == 1)
            {
                ranges = np.repeat(ranges, n_var).reshape(2, n_var).T;
            }

            NDArray train_input = np.zeros(new int[] { train_num, n_var });
            NDArray test_input = np.zeros(new int[] { test_num, n_var });
            for (int i = 0; i < n_var; ++i)
            {
                train_input[Slice.All, Slice.Index(i)] = np.random.rand(train_num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0];
                test_input[Slice.All, Slice.Index(i)] = np.random.rand(test_num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0];
            }

            Tensor train_inputT = torch.frombuffer(train_input.ToArray<double>(), dtype:torch.float64).view(new long[] { train_num, n_var });
            Tensor test_inputT = torch.frombuffer(test_input.ToArray<double>(), dtype: torch.float64).view(new long[] { test_num, n_var });
            Tensor train_label = f(train_inputT);
            Tensor test_label = f(test_inputT);

            Tensor normalize(Tensor data, Tensor mean, Tensor std)
            {
                return (data - mean) / std;
            }

            if (normalize_input)
            {
                Tensor mean_input = torch.mean(train_inputT, dimensions:  new long[] { 0 }, keepdim: true);
                Tensor std_input = torch.std(train_inputT, dimensions: new long[] { 0 }, keepdim: true);
                train_inputT = normalize(train_inputT, mean_input, std_input);
                test_inputT = normalize(test_inputT, mean_input, std_input);
            }

            if (normalize_label)
            {
                Tensor mean_label = torch.mean(train_label, dimensions: new long[] { 0 }, keepdim: true);
                Tensor std_label = torch.std(train_label, dimensions: new long[] { 0 }, keepdim: true);
                train_label = normalize(train_label, mean_label, std_label);
                test_label = normalize(test_label, mean_label, std_label);
            }

            Dictionary<string, Tensor> dataset = new Dictionary<string, Tensor>()
            {
                {"train_input", train_inputT.to(device)},
                {"test_input", test_inputT.to(device)},
                {"train_label", train_label.to(device)},
                {"test_label", test_label.to(device)}
            };

            return dataset;
        }

        public static Tuple<Tensor, double> fit_params(Tensor x,  Tensor y, Func<Tensor, Tensor> fun, NDArray a_range = null, NDArray b_range = null,
                                                            int grid_number = 101, int iterations = 3, bool verbose = true, string device = "cpu", bool train_LR_model = false)
        {
            torch.set_default_dtype(torch.float64);

            if (a_range == null)
            {
                a_range = new NDArray(values: new double[] { -10.0, 10.0 });
            }

            if (b_range == null)
            {
                b_range = new NDArray(values: new double[] { -10.0, 10.0 });
            }

            Tensor y_mean = torch.mean(y, dimensions : new long[] { 0 }, type: torch.float64, keepdim : true);
            Tensor y_mean_diff = (y - y_mean)[TensorIndex.Colon, TensorIndex.None, TensorIndex.None];
            Tensor y_mean_MSE = torch.sum(torch.square(y_mean_diff), type: float64, dim : 0);

            Tensor a_ = torch.zeros(1), b_ = torch.zeros(1), a_grid, b_grid, post_fun, x_mean, numerator, denominator, r2 = torch.zeros(1);
            long a_id = 0, b_id = 0;
            for (int _  = 0; _ < iterations; ++_)
            {
                a_ = torch.linspace(a_range[0], a_range[1], steps: grid_number, dtype: torch.float64, device: device);
                b_ = torch.linspace(b_range[0], b_range[1], steps: grid_number, dtype: torch.float64, device: device);
                Tensor[] meshgrid = torch.meshgrid( tensors: new Tensor[]{a_, b_});
                a_grid = meshgrid[0];
                b_grid = meshgrid[1];
                post_fun = fun(a_grid[TensorIndex.None, TensorIndex.Colon, TensorIndex.Colon] * 
                            x[TensorIndex.Colon, TensorIndex.None, TensorIndex.None] + 
                            b_grid[TensorIndex.None, TensorIndex.Colon, TensorIndex.Colon]);
                x_mean = torch.mean(post_fun, dimensions : new long[] { 0 }, type: torch.float64, keepdim: true);
                numerator = torch.square(torch.sum((post_fun - x_mean) * y_mean_diff, type: torch.float64, dim: 0));
                denominator = torch.sum(torch.square(post_fun - x_mean), type: torch.float64, dim: 0) * y_mean_MSE;
                r2 = torch.nan_to_num(numerator / (denominator + 1e-4));

                long best_id = torch.argmax(r2).ToInt64();
                a_id = torch.div(best_id, grid_number, rounding_mode: RoundingMode.floor).ToInt64();
                b_id = best_id % grid_number;
                double[] a_data = a_.data<double>().ToArray();
                double[] b_data = b_.data<double>().ToArray();

                if (a_id == 0 || a_id == grid_number - 1 || b_id == 0 || b_id == grid_number - 1)
                {
                    if (_ == 0 && verbose)
                    {
                        Console.WriteLine("Best value at boundary.");
                    }
                    if(a_id == 0)
                    {
                        a_range = new NDArray(values: new double[] { a_data[0], a_data[1] });
                    }
                    if(a_id ==  grid_number - 1)
                    {
                        a_range = new NDArray(values: new double[] { a_data[a_data.Length - 2], a_data[a_data.Length - 1] });
                    }
                    if (b_id == 0)
                    {
                        b_range = new NDArray(values: new double[] { b_data[b_data.Length - 2], b_data[b_data.Length - 1] });
                    }
                    if (b_id == grid_number - 1)
                    {
                        b_range = new NDArray(values: new double[] { b_data[b_data.Length - 2], b_data[b_data.Length - 1] });
                    }
                }
                else
                {
                    a_range = new NDArray(values: new double[] { a_data[a_id - 1], a_data[a_id + 1] });
                    b_range = new NDArray(values: new double[] { b_data[b_id - 1], b_data[b_id + 1] });
                }
            }

            double a_best = a_[a_id].ToDouble(), b_best = b_[b_id].ToDouble();
            post_fun = fun(a_best * x + b_best);
            double r2_best = r2[a_id, b_id].ToDouble();

            if (verbose)
            {
                Console.WriteLine($"r2 is {r2_best}");
                if (r2_best < 0.9)
                {
                    Console.WriteLine("r2 is not very high, please double check if you are choosing the correct symbolic function.");
                }
            }

            double[] coeffs;
            if (train_LR_model)
            {
                post_fun = torch.nan_to_num(post_fun);
                coeffs = fit_LR(post_fun.detach().unsqueeze_(1), y.detach().unsqueeze_(1));
            }
            else
            {
                post_fun = torch.stack(new Tensor[] { torch.nan_to_num(post_fun), torch.ones_like(post_fun) }, dim: 1).squeeze();
                coeffs = torch.linalg.lstsq(post_fun, y).Solution.data<double>().ToArray();
            }

            Tensor c_best = coeffs[0];
            Tensor d_best = coeffs[1];
            return new Tuple<Tensor, double>(torch.stack(new Tensor[] { a_best, b_best, c_best, d_best }), r2_best);
        }

        private static double[] fit_LR(Tensor x, Tensor y)
        {
            Linear_Regression model = new Linear_Regression();
            
            Linear_Regression best_model = model;
            double best_loss = 100;

            Loss<torch.Tensor, torch.Tensor, torch.Tensor> criterion = new MSELoss();
            optim.Optimizer optimizer = new SGD(model.parameters(), lr: 1e-2);
            int num_epochs = 700;

            for(int epoch = 0; epoch < num_epochs; ++epoch)
            {
                optimizer.zero_grad();

                Tensor output = model.forward(x);
                Tensor loss = criterion.forward(output, y);

                if(best_loss - loss.ToDouble() > 1e-5)
                {
                    best_loss = loss.ToDouble();
                    best_model = model;
                }

                loss.backward();
                optimizer.step();
            }

            return best_model.get_params();
        }

        //static int Main()
        //{
        //    torch.set_default_dtype(torch.float64);

        //    int num = 100;
        //    Tensor x = torch.linspace(-1.0, 1.0, steps: num);
        //    Tensor noises = torch.normal(0, 1, new long[] { num }) * 0.02;
        //    Tensor y = 5.0 * torch.sin(3.0 * x + 2.0) + 0.7 + noises;

        //    Tuple<Tensor, double> parameters = fit_params(x, y, torch.sin, train_LR_model: false);
        //    Console.WriteLine(parameters.Item1.ToString(style: TorchSharp.TensorStringStyle.Numpy) + " " + parameters.Item2);

        //    //Func<Tensor, Tensor> f = (Tensor x) => torch.exp(torch.sin(np.pi * x[TensorIndex.Colon, TensorIndex.Slice(0, 1)]) +
        //    //                                        torch.square(x[TensorIndex.Colon, TensorIndex.Slice(1, 2)]));

        //    //Dictionary<string, Tensor> dataset = create_dataset(f, n_var: 2, train_num: 100);
        //    //Console.WriteLine(dataset["train_input"]);
        //    return 0;
        //}
    }
}
