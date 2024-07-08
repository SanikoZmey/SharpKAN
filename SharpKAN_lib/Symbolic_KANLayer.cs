using System;
using NumSharp;
using TorchSharp;
using TorchSharp.Modules;
using static SharpKAN_lib.Utils;
using static TorchSharp.torch;

namespace SharpKAN_lib
{
    public class Symbolic_KANLayer : nn.Module<Tensor, Tuple<Tensor, Tensor>>
    {
        public int in_dim, out_dim;
        string device;

        public Parameter mask, affine;
        public string[][] funs_names;
        private Func<Tensor, Tensor>[][] funs;
        public Symbolic_KANLayer(int in_dim = 3, int out_dim = 2, string device = "cpu") : base(nameof(Symbolic_KANLayer))
        {
            torch.set_default_dtype(torch.float64);

            this.in_dim = in_dim;
            this.out_dim = out_dim;
            this.device = device;

            mask = torch.nn.Parameter(torch.zeros(out_dim, in_dim, device: this.device), requires_grad: false);
            funs_names = new string[out_dim][];
            funs = new Func<Tensor, Tensor>[out_dim][];
            for(int j =  0; j < out_dim; ++j)
            {
                funs_names[j] = new string[in_dim];
                funs[j] = new Func<Tensor, Tensor>[in_dim];
                for (int i = 0; i < in_dim; ++i)
                {
                    funs_names[j][i] = "";
                    funs[j][i] = (Tensor x) => x;
                }
            }
            this.affine = torch.nn.Parameter(torch.zeros(out_dim, in_dim, 4, device: this.device));

            RegisterComponents();
        }

        public override Tuple<Tensor, Tensor> forward(Tensor x)
        {
            torch.set_default_dtype(torch.float64);

            Tensor[] postacts = new Tensor[in_dim];
            for (int i = 0; i < in_dim; ++i)
            {
                Tensor[] postacts_ = new Tensor[out_dim];
                for(int j = 0; j < out_dim; ++j)
                {
                    Tensor xj = affine[j, i, 2] * funs[j][i](affine[j, i, 0] * x[TensorIndex.Colon, TensorIndex.Slice(i, i + 1)] + affine[j, i, 1]) + affine[j, i, 3];
                    postacts_[j] = xj * mask[j][i];
                }
                postacts[i] = torch.stack(postacts_);
            }
            
            Tensor postactsT = torch.stack(postacts).permute(2, 1, 0, 3)[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, 0];
            Tensor y = torch.sum(postactsT, type: torch.float64, dim: 2);

            return new Tuple<Tensor, Tensor>(y, postactsT);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
        }

        public Symbolic_KANLayer get_subset(long[] in_id, long[] out_id)
        {
            Symbolic_KANLayer sbb = new Symbolic_KANLayer(in_id.Length, out_id.Length, device: device);

            sbb.mask.set_(mask[TensorIndex.Tensor(out_id)][TensorIndex.Colon, TensorIndex.Tensor(in_id)]);
            int row_count = 0;
            foreach(int j in out_id)
            {
                int col_count = 0;
                foreach(int i in in_id)
                {
                    sbb.funs[row_count][col_count] = funs[j][i];
                    sbb.funs_names[row_count][col_count++] = funs_names[j][i];
                }
                row_count++;
            }
            sbb.affine.requires_grad_(false).set_(affine[TensorIndex.Tensor(out_id)][TensorIndex.Colon, TensorIndex.Tensor(in_id)]).requires_grad_(true);

            return sbb;
        }

        public double fix_symbolic(int i, int j, string fun_name, Tensor x = null, Tensor y = null, bool random = false,
                                    NDArray a_range = null, NDArray b_range = null, bool verbose = true)
        {
            torch.set_default_dtype(torch.float64);

            if (SYMBOLIC_LIB.ContainsKey(fun_name))
            {
                Func<Tensor, Tensor> fun = SYMBOLIC_LIB[fun_name];
                funs_names[j][i] = fun_name;
                funs[j][i] = fun;

                if(x is null || y is null)
                {
                    if(random)
                    {
                        affine.requires_grad_(false)[j][i] = torch.rand(4, dtype: torch.float64) * 2 - 1;
                        affine.requires_grad_(true);
                    }
                    else
                    {
                        affine.requires_grad_(false)[j][i] = torch.tensor(new double[] { 1, 0, 1, 0 });
                        affine.requires_grad_(true);
                    }
                    return double.NaN;
                }

                Tuple<Tensor, double>  result = fit_params(x, y, fun, a_range, b_range, verbose: verbose, device: device);
                affine.requires_grad_(false)[j][i] = result.Item1;
                affine.requires_grad_(true);

                return result.Item2;
            }
            
            if(verbose)
            {
                Console.WriteLine($"There is no such function \"{fun_name}\" in functions library. Please check the correctness of the name or pass the function itself.");
            }
            return double.NaN;
        }

        public double fix_symbolic(int i, int j, Func<Tensor, Tensor> func, Tensor x = null, Tensor y = null, bool random = false,
                                    NDArray a_range = null, NDArray b_range = null, bool verbose = true)
        {
            torch.set_default_dtype(torch.float64);

            funs_names[j][i] = "user_specified";
            funs[j][i] = func;

            if (x is null || y is null)
            {
                if (random)
                {
                    affine.requires_grad_(false)[j][i] = torch.rand(4, dtype: torch.float64) * 2 - 1;
                    affine.requires_grad_(true);
                }
                else
                {
                    affine.requires_grad_(false)[j][i] = torch.tensor(new double[] { 1, 0, 1, 0 });
                    affine.requires_grad_(true);
                }
                return double.NaN;
            }

            Tuple<Tensor, double> result = fit_params(x, y, func, a_range, b_range, verbose: verbose, device: device);
            affine.requires_grad_(false)[j][i] = result.Item1;
            affine.requires_grad_(true);

            return result.Item2;
        }
    }

    //public class main
    //{
    //    static int Main()
    //    {
    //        torch.set_default_dtype(torch.float64);

    //        Symbolic_KANLayer sb = new Symbolic_KANLayer(in_dim: 2, out_dim: 2);
    //        int counter = 0;
    //        foreach (Parameter parameter in sb.parameters())
    //        {
    //            Console.WriteLine(parameter + " " + counter++);
    //        }

    //        //int batch_size = 100;
    //        //Tensor x = torch.linspace(-1, 1, steps: batch_size);
    //        //Tensor noises = torch.normal(0, 1, new long[] { batch_size }) * 0.02;
    //        //Tensor y = 5.0 * torch.sin(3.0 * x + 2.0) + 0.7 + noises;
    //        //double r2 = sb.fix_symbolic(2, 1, "sin", x, y);
    //        //for (int i = 0; i < sb.funs_names.Length; ++i)
    //        //{
    //        //    foreach (string name in sb.funs_names[i])
    //        //    {
    //        //        Console.WriteLine(name == "" ? "- " : name);
    //        //    }
    //        //}
    //        //Console.WriteLine(sb.affine[1, 2, TensorIndex.Colon].ToString(style: TorchSharp.TensorStringStyle.Numpy));

    //        //Symbolic_KANLayer sb = new Symbolic_KANLayer(in_dim: 3, out_dim: 5);
    //        //Tensor x = torch.normal(0, 1, size: new long[] { 100, 3 });
    //        //Tuple<Tensor, Tensor> result = sb.forward(x);
    //        //Console.WriteLine(result.Item1 + "\n" + result.Item2);
    //        //Console.WriteLine(result.Item1.ToString(style: TorchSharp.TensorStringStyle.Numpy) + "\n" + result.Item2.ToString(style: TorchSharp.TensorStringStyle.Numpy));

    //        //Symbolic_KANLayer sb_large = new Symbolic_KANLayer(in_dim: 10, out_dim: 10);
    //        //Symbolic_KANLayer sb_small = sb_large.get_subset(new long[] { 0, 9 }, new long[] { 1, 2, 3 });
    //        //Console.WriteLine(sb_small.in_dim + " " + sb_small.out_dim);

    //        return 0;
    //    }
    //}
}
