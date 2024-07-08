using System;
using System.Collections.Generic;
using NumSharp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using TorchSharp.Modules;
using ShellProgressBar;
using System.Linq;
using NumSharp.Utilities;
using System.IO;
using System.Runtime.InteropServices;

namespace SharpKAN_lib
{
    public class KAN : nn.Module<Tensor, Tensor>
    {

        private int depth, grid, k;
        private string device;
        private int[] width;
        private bool symbolic_enabled, bias_trainable;

        public ModuleList<KANLayer> act_funs;
        public ModuleList<Symbolic_KANLayer> symbolic_funs;
        private ModuleList<Linear> biases;
        private torch.nn.Module<torch.Tensor, torch.Tensor> base_fun;

        private Tensor[] acts, spline_preacts, spline_postacts, spline_postsplines, acts_scale, acts_scale_std, mask;


        public KAN(int[] width = null, int grid = 3, int k = 3, double noise_scale = 0.1, double noise_scale_base = 1,
                        torch.nn.Module<torch.Tensor, torch.Tensor> base_fun = null, bool symbolic_enabled = true, bool bias_trainable = true,
                        double grid_eps = 1.0, NDArray grid_range = null, bool sp_trainable = true, bool sb_trainable = true, string device = "cpu", int seed = 0) : base(nameof(KAN))
        {
            torch.manual_seed(seed);
            np.random.seed(seed);

            torch.set_default_dtype(torch.float64);

            if (base_fun == null)
            {
                base_fun = nn.SiLU();
            }

            if (grid_range == null)
            {
                grid_range = new NDArray(new double[] { -1.0, 1.0 });
            }

            this.device = device;
            this.depth = width.Length - 1;
            this.width = width;
            this.grid = grid;
            this.k = k;
            this.base_fun = base_fun;
            this.bias_trainable = bias_trainable;

            Linear[] biases = new Linear[this.depth];
            KANLayer[] act_funs = new KANLayer[this.depth];
            Symbolic_KANLayer[] symbolic_funs = new Symbolic_KANLayer[this.depth];


            for (int l = 0; l < this.depth; ++l)
            {
                Tensor scale_base = 1 / torch.sqrt(width[l]) + (torch.randn(width[l] * width[l + 1], dtype: torch.float64) * 2 - 1) * noise_scale_base;
                act_funs[l] = new KANLayer(in_dim: width[l], out_dim: width[l + 1], num: this.grid, k: this.k, noise_scale: noise_scale, scale_base: scale_base,
                                                    scale_sp: 1, base_fun: base_fun, grid_eps: grid_eps, grid_range: grid_range, sp_trainable: sp_trainable,
                                                    sb_trainable: sb_trainable, device: this.device);

                symbolic_funs[l] = new Symbolic_KANLayer(in_dim: width[l], out_dim: width[l + 1], device: this.device);

                Linear bias = nn.Linear(width[l + 1], 1, hasBias: false, dtype: torch.float64, device: this.device);
                bias.weight.requires_grad_(false).set_(torch.zeros_like(bias.weight)).requires_grad_(this.bias_trainable);
                biases[l] = bias;
            }

            this.biases = new ModuleList<Linear>(biases);
            this.act_funs = new ModuleList<KANLayer>(act_funs);
            this.symbolic_funs = new ModuleList<Symbolic_KANLayer>(symbolic_funs);
            this.symbolic_enabled = symbolic_enabled;

            RegisterComponents();
        }
        public override Tensor forward(Tensor x)
        {
            torch.set_default_dtype(torch.float64);

            acts = new Tensor[depth + 1];
            spline_preacts = new Tensor[depth];
            spline_postacts = new Tensor[depth];
            spline_postsplines = new Tensor[depth];
            acts_scale = new Tensor[depth];
            acts_scale_std = new Tensor[depth];

            acts[0] = x;

            Tuple<Tensor, Tensor, Tensor, Tensor> act_result;
            Tuple<Tensor, Tensor> symb_result;
            Tensor postacts, grid_reshape;

            for (int l = 0; l < depth; ++l)
            {
                act_result = act_funs[l].forward(x);

                //Console.WriteLine(act_result.Item1.ToString(style: TorchSharp.TensorStringStyle.Numpy));

                if (symbolic_enabled)
                {
                    symb_result = symbolic_funs[l].forward(x);
                }
                else
                {
                    symb_result = new Tuple<Tensor, Tensor>(torch.zeros(1), torch.zeros(1));
                }

                x = act_result.Item1 + symb_result.Item1;
                postacts = act_result.Item3 + symb_result.Item2;

                grid_reshape = act_funs[l].grid.reshape(width[l + 1], width[l], -1);
                acts_scale[l] = torch.mean(torch.abs(postacts), dimensions: new long[] { 0 }, type: torch.float64) /
                                            (grid_reshape[TensorIndex.Colon, TensorIndex.Colon, -1] - grid_reshape[TensorIndex.Colon, TensorIndex.Colon, 0] + 1e-4);
                acts_scale_std[l] = torch.std(postacts, type: torch.float64, dimensions: 0);
                spline_preacts[l] = act_result.Item2.detach();
                spline_postacts[l] = postacts.detach();
                spline_postsplines[l] = act_result.Item4.detach();

                x += biases[l].weight;
                acts[l + 1] = x;
            }
            return x;
        }
        public void update_grid_from_samples(Tensor x)
        {
            using (torch.no_grad())
            {
                forward(x);
                for (int l = 0; l < depth; ++l)
                {
                    act_funs[l].update_grid_from_samples(acts[l]);
                }
            }
            
        }

        public void initialize_grid_from_another_model(KAN model, Tensor x)
        {
            model.forward(x);
            for (int l = 0; l < depth; ++l)
            {
                act_funs[l].initialize_grid_from_parent(model.act_funs[l], model.acts[l]);
            }
        }

        KAN initialize_from_another_model(KAN another_model, Tensor x)
        {
            //another_model.forward(x);
            long batch_size = x.shape[0];

            initialize_grid_from_another_model(another_model, x.to(another_model.device));

            for (int l = 0; l < depth; ++l)
            {
                KANLayer spb = act_funs[l];
                KANLayer spb_parent = another_model.act_funs[l];

                Tensor preacts = another_model.spline_preacts[l];
                Tensor postsplines = another_model.spline_postsplines[l];
                act_funs[l].coef.requires_grad_(false).set_(Spline.curve2coeff(preacts.reshape(batch_size, spb.size).permute(1, 0),
                                                                                postsplines.reshape(batch_size, spb.size).permute(1, 0),
                                                                                spb.grid, k: spb.k, device: device)).requires_grad_(true);
                spb.scale_base.requires_grad_(false).set_(spb_parent.scale_base).requires_grad_(spb.sb_trainable);
                spb.scale_sp.requires_grad_(false).set_(spb_parent.scale_sp).requires_grad_(spb.sp_trainable);
                spb.mask.set_(spb_parent.mask);

                biases[l].weight.requires_grad_(false).set_(another_model.biases[l].weight).requires_grad_(bias_trainable);
                symbolic_funs[l] = another_model.symbolic_funs[l];
            }

            return this;
        }

        public void set_mode(int l, int i, int j, string mode, double mask_n = double.NaN)
        {
            double mask_s;
            switch (mode)
            {
                case "s":
                    mask_n = 0;
                    mask_s = 1;
                    break;
                case "n":
                    mask_n = 1;
                    mask_s = 0;
                    break;
                case "ns":
                    if (mask_n is double.NaN)
                    {
                        mask_n = 1;
                    }
                    mask_s = 1;
                    break;
                case "sn":
                    if (mask_n is double.NaN)
                    {
                        mask_n = 1;
                    }
                    mask_s = 1;
                    break;
                default:
                    mask_n = mask_s = 0;
                    break;
            }

            act_funs[l].mask[j * act_funs[l].in_dim + i] = mask_n;
            symbolic_funs[l].mask[j, i] = mask_s;
        }

        public double fix_symbolic(int l, int i, int j, string fun_name, bool fit_parameters = true,
                                    NDArray a_range = null, NDArray b_range = null, bool verbose = true, bool random = false)
        {
            set_mode(l, i, j, mode: "s");
            if (!fit_parameters)
            {
                symbolic_funs[l].fix_symbolic(i, j, fun_name, verbose: verbose, random: random);
                return double.NaN;
            }

            Tensor x = acts[l][TensorIndex.Colon, i];
            Tensor y = spline_postacts[l][TensorIndex.Colon, j, i];

            return symbolic_funs[l].fix_symbolic(i, j, fun_name, x, y, a_range: a_range, b_range: b_range, verbose: verbose);
        }

        public void unfix_symbolic(int l, int i, int j)
        {
            set_mode(l, i, j, "n");
        }

        public void unfix_symbolic_all()
        {
            for (int l = 0; l < depth; ++l)
            {
                for (int i = 0; i < width[l]; ++i)
                {
                    for (int j = 0; j < width[l + 1]; ++j)
                    {
                        unfix_symbolic(l, i, j);
                    }
                }
            }
        }

        public void lock_(int l, NDArray ids)
        {
            act_funs[l].lock_(ids);
        }

        public void unlock_(int l, NDArray ids)
        {
            act_funs[l].unlock_(ids);
        }

        public Tuple<double, double, double, double> get_range(int l, int i, int j, bool verbose = true)
        {
            Tensor x = spline_preacts[l][TensorIndex.Colon, j, i];
            Tensor y = spline_postacts[l][TensorIndex.Colon, j, i];

            double x_min, x_max, y_min, y_max;
            x_min = x.min().ToDouble();
            x_max = x.max().ToDouble();
            y_min = y.min().ToDouble();
            y_max = y.max().ToDouble();

            if(verbose)
            {
                Console.WriteLine($"x range: [{x_min:f2}, {x_max:f2}]");
                Console.WriteLine($"y range: [{y_min:f2}, {y_max:f2}]");
            }

            return new Tuple<double, double, double, double>(x_min, x_max, y_min, y_max);
        }

        public Dictionary<string, double[]> train(Dictionary<string, Tensor> dataset, string optim_name = "LBFGS", int steps = 100, int logging_freq = 1, 
                                                    double lambda = 0, double lambda_l1 = 1, double lambda_entropy = 2, double lambda_coeff = 0, 
                                                    double lambda_coeffdiff = 0, bool update_grid = true, int grid_update_num = 10, int stop_grid_update_step = 50, 
                                                    Func<Tensor, Tensor, Tensor> loss_fn = null, double lr = 1, long batch_size = -1, 
                                                    double small_mag_threshold = 1e-16, double small_reg_factor = 1, bool sglr_avoid = false)
        {
            torch.set_default_dtype(torch.float64);

            Tensor reg(Tensor[] acts_scale)
            {
                Tensor nonlinear(Tensor x, double th = double.NaN, double factor = double.NaN)
                {
                    if(th is double.NaN)
                    {
                        th = small_mag_threshold;
                    }
                    if(factor is double.NaN)
                    {
                        factor = small_reg_factor;
                    }
                    Tensor res = x.lt(th) * x * factor + x.gt(th) * (x + (factor - 1.0) * th);
                    return res;
                }

                Tensor reg_ = torch.zeros(1), vec, p, l1, entropy, coeff_l1, coeff_diff_l1;
                for(int i = 0; i < acts_scale.Length; ++i)
                {
                    vec = acts_scale[i].reshape(-1, 1);
                    p = vec / torch.sum(vec, type: torch.float64, dim: 0);
                    l1 = torch.sum(nonlinear(vec), type: torch.float64, dim: 0);
                    entropy = - torch.sum(p * torch.log2(p + 1e-4), type: torch.float64, dim: 0);
                    reg_ += (lambda_l1 * l1 + lambda_entropy * entropy);

                    Console.WriteLine(vec.ToString(style: TorchSharp.TensorStringStyle.Numpy) + "\n" +
                                       p.ToString(style: TorchSharp.TensorStringStyle.Numpy) + "\n" +
                                       l1.ToString(style: TorchSharp.TensorStringStyle.Numpy) + "\n" +
                                       entropy.ToString(style: TorchSharp.TensorStringStyle.Numpy) + "\n" +
                                       reg_.ToString(style: TorchSharp.TensorStringStyle.Numpy));
                }

                for(int i = 0; i < act_funs.Count; ++i)
                {
                    coeff_l1 = torch.sum(torch.mean(torch.abs(act_funs[i].coef), type: torch.float64, dimensions: new long[] { 1 }), type: torch.float64, dim: 0);
                    coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(act_funs[i].coef)), type: torch.float64, dimensions: new long[] { 1 }), type: torch.float64, dim: 0);
                    reg_ += (lambda_coeff * coeff_l1 + lambda_coeffdiff * coeff_diff_l1);
                }

                return reg_;
            }

            if(loss_fn == null)
            {
                loss_fn = (Tensor x, Tensor y) => torch.mean(torch.square(x - y), type: torch.float64, dimensions: new long[] { 0 });
            }

            optim.Optimizer optimizer = null;
            switch(optim_name)
            {
                case "Adam":
                    optimizer = Adam(parameters(), lr);
                    break;
                default:
                    optimizer = LBFGS(parameters(), lr: lr, history_size: 10, tolerange_grad: 1e-32, tolerance_change: 1e-32);
                    break;
            }

            Dictionary<string, double[]> results = new Dictionary<string, double[]>();
            results["train_loss"] = new double[steps];
            results["test_loss"] = new double[steps];
            results["reg"] = new double[steps];

            long batch_size_test = batch_size;
            if(batch_size == -1 || batch_size > dataset["train_input"].shape[0])
            {
                batch_size = dataset["train_input"].shape[0];
                batch_size_test = dataset["test_input"].shape[0];
            }

            int grid_update_freq = stop_grid_update_step / grid_update_num;

            Tensor train_id, test_id, pred, id_, train_loss = torch.zeros(1), test_loss = torch.zeros(1), _reg = torch.ones(1);

            Tensor closure()
            {
                optimizer.zero_grad();
                //Console.WriteLine(dataset["train_label"][train_id].ToString(style: TorchSharp.TensorStringStyle.Numpy));
                pred = forward(dataset["train_input"][train_id].to(device));
                if(sglr_avoid)
                {
                    id_ = torch.where(torch.isnan(torch.sum(pred, type: torch.float64, dim: 1)) == false)[0];
                    train_loss = loss_fn(pred[id_], dataset["train_label"][train_id][id_].to(device));
                }
                else
                {
                    train_loss = loss_fn(pred, dataset["train_label"][train_id].to(device));
                }
                //Console.WriteLine(pred.ToString(style: TorchSharp.TensorStringStyle.Numpy));
                _reg = reg(acts_scale);
                Tensor objective = train_loss + lambda * _reg;

                Console.WriteLine(objective.ToString(style: TorchSharp.TensorStringStyle.Numpy));
                Console.WriteLine(_reg.ToDouble() * lambda);

                bool again = false;
                for(int i = 0; i < 10; ++i)
                {
                    try
                    {
                        objective.backward();
                    }
                    catch (ExternalException ex)
                    {
                        Console.WriteLine($"{ex.Message}");
                    }

                    if(!again)
                    {
                        break;
                    }
                }                

                return objective;
            }


            //using (var pbar = new ProgressBar((int)Math.Ceiling((double)steps / logging_freq), "Starting training...", new ProgressBarOptions { ProgressBarOnBottom = true }))
            //{
                for (int _ = 0; _ < steps; ++_)
                {
                    train_id = torch.frombuffer(np.random.choice(np.arange(dataset["train_input"].shape[0]), new int[] { (int)batch_size }, replace: false).ToArray<float>(), dtype: torch.int64);
                    test_id = torch.frombuffer(np.random.choice(np.arange(dataset["test_input"].shape[0]), new int[] { (int)batch_size_test }, replace: false).ToArray<float>(), dtype: torch.int64);

                    //Console.WriteLine(train_id.ToString(style: TorchSharp.TensorStringStyle.Numpy));

                    if (_ % grid_update_freq == 0 && _ < stop_grid_update_step && update_grid)
                    {
                        update_grid_from_samples(dataset["train_input"][train_id].to(device));
                    }

                    optimizer.step(closure);

                    test_loss = loss_fn(forward(dataset["test_input"][test_id].to(device)), dataset["test_label"][test_id].to(device));

                    //if (_ % logging_freq == 0)
                    //{
                    //    pbar.Tick($"Train loss: {train_loss.sqrt().detach().cpu().ToDouble():e2} | Test loss: {test_loss.detach().cpu().ToDouble():e2} | Reg. : {_reg.detach().cpu().ToDouble():e2}");
                    //}

                    results["train_loss"][_] = torch.sqrt(train_loss).detach().cpu().ToDouble();
                    results["test_loss"][_] = torch.sqrt(test_loss).detach().cpu().ToDouble();
                    results["reg"][_] = _reg.detach().cpu().ToDouble();
                }
            //}

            return results;
        }

        public void remove_node(int l, int i)
        {
            torch.set_default_dtype(torch.float64);

            act_funs[l - 1].mask[i * width[l - 1] + torch.arange(width[l - 1], dtype: torch.float64)] = 0;
            act_funs[l].mask[torch.arange(width[l + 1], dtype: torch.float64) * width[l] + i] = 0;
            symbolic_funs[l - 1].mask[i, TensorIndex.Colon] = torch.zeros_like(symbolic_funs[l - 1].mask[i, TensorIndex.Colon]);
            symbolic_funs[l].mask[TensorIndex.Colon, i] = torch.zeros_like(symbolic_funs[l].mask[TensorIndex.Colon, i]);
        }

        public void remove_edge(int l, int i, int j)
        {
            act_funs[l].mask[j * width[l] + i] = 0;
        }

        public KAN prune(double threshold, string mode = "string", NDArray active_neurons_id = null)
        {
            torch.set_default_dtype(torch.float64);

            Tensor[] mask = new Tensor[acts_scale.Length + 2];
            long[][] active_neurons = new long[acts_scale.Length + 2][];

            mask[0] = torch.ones(width[0]);
            active_neurons[0] = np.arange(width[0]).ToArray<long>();

            Tensor overall_important = torch.zeros(1), in_important, out_important;
            for (int i = 0; i < acts_scale.Length - 1; ++i)
            {
                switch(mode)
                {
                    case "auto":
                        in_important = torch.max(acts_scale[i], dim: 1).values > threshold;
                        out_important = torch.max(acts_scale[i + 1], dim: 1).values > threshold;
                        overall_important = in_important * out_important;
                        break;
                    case "manual":
                        overall_important = torch.zeros(width[i + 1], dtype: torch.@bool);
                        overall_important[torch.frombuffer(active_neurons_id[i + 1].ToArray<long>(), dtype: torch.int64)] = true;
                        break;
                }
                mask[i] = overall_important.clone();
                active_neurons[i] = torch.where((overall_important == true))[0].data<long>().ToArray();
            }
            mask[acts_scale.Length + 1] = torch.ones(width[width.Length - 1]);
            active_neurons[acts_scale.Length + 1] = np.arange(width[width.Length - 1]).ToArray<long>();

            this.mask = mask;

            for (int l = 0; l < acts_scale.Length - 1; ++l)
            {
                for(int i = 0; i < width[l + 1]; ++i)
                {
                    if (!active_neurons[l + 1].Contains(i))
                    {
                        remove_node(l + 1, i);
                    }

                }
            }

            KAN model2 = new KAN(width.CloneArray(), grid, k, base_fun: base_fun, device: device);
            model2.load_state_dict(state_dict());
            for(int i = 0; i <= acts_scale.Length; ++i)
            {
                if(i < acts_scale.Length - 1)
                {
                    model2.biases[i].weight.requires_grad_(false).set_(
                        model2.biases[i].weight[TensorIndex.Colon, TensorIndex.Tensor(active_neurons[i + 1])]).requires_grad_(bias_trainable);
                }
                model2.act_funs[i] = model2.act_funs[i].get_subset(active_neurons[i], active_neurons[i + 1]);
                model2.width[i] = active_neurons[i].Length;
                model2.symbolic_funs[i] = symbolic_funs[i].get_subset(active_neurons[i], active_neurons[i + 1]);
            }

            return model2;
        }

        public Tuple<string, Func<Tensor, Tensor>, double> suggest_symbolic(int l, int i, int j, NDArray a_range = null, NDArray b_range = null, Dictionary<string, Func<Tensor, Tensor>> lib = null,
                                        int topk = 5, bool verbose = true)
        {
            Dictionary<string, Func<Tensor, Tensor>> symbolic_lib = new Dictionary<string, Func<Tensor, Tensor>>();
            if (lib == null)
            {
                symbolic_lib = Utils.SYMBOLIC_LIB;
            }
            else
            {
                foreach(string func_name in lib.Keys)
                {
                    symbolic_lib[func_name] = lib[func_name];
                }
            }

            double[] r2s = new double[symbolic_lib.Count];
            int counter = 0;

            foreach(string func_name in symbolic_lib.Keys)
            {
                r2s[counter++] = fix_symbolic(l, i, j, func_name, a_range : a_range, b_range: b_range, verbose : false);
            }

            unfix_symbolic(l, i, j);

            topk = np.minimum(topk, symbolic_lib.Count);
            int[] sorted_ids = np.argsort<double>(r2s)["::-1"][$":{topk}"].ToArray<int>();
            if(verbose)
            {
                Console.WriteLine("function , r2");
                for (int iter = 0; iter < topk; ++iter)
                {
                    Console.WriteLine(symbolic_lib.Keys.ElementAt(sorted_ids[iter]) + " , " + r2s[sorted_ids[iter]]);
                }
            }

            return new Tuple<string, Func<Tensor, Tensor>, double>(symbolic_lib.Keys.ElementAt(0), symbolic_lib.Values.ElementAt(0), r2s[0]);
        }

        public void auto_symbolic(NDArray a_range = null, NDArray b_range = null, Dictionary<string, Func<Tensor, Tensor>> lib = null, bool verbose = true)
        {
            for (int l = 0; l < depth; ++l)
            {
                for(int i = 0; i < width[l]; ++i)
                {
                    for(int j = 0; j < width[l + 1]; ++j)
                    {
                        if(symbolic_funs[l].mask[j, i].ToInt32() > 0)
                        {
                            Console.WriteLine($"Skipping ({l}, {i}, {j}) since already symbolic");
                        }
                        else
                        {
                            Tuple<string, Func<Tensor, Tensor>, double> suggested_funcs = suggest_symbolic(l, i, j, a_range: a_range, b_range: b_range,
                                                                                                            lib: lib, verbose: false);
                            fix_symbolic(l, i, j, fun_name: suggested_funcs.Item1, verbose: verbose);
                            if(verbose)
                            {
                                Console.WriteLine($"Fixing ({l},{i},{j}) with {suggested_funcs.Item1}, r2 = {suggested_funcs.Item3}");
                            }
                        }
                    }
                }
            }
        }

        /// TODO: Somehow implement func2string algo...
        /// 
        //public string symbolic_formula(int floating_points, string[] var, NDArray normals = null, NDArray output_normals = null)
        //{

        //}

        public void clear_ckpts(string folder = "./model_ckpts")
        {
            if (Directory.Exists(folder))
            {
                DirectoryInfo di = new DirectoryInfo(folder);
                foreach (FileInfo file in di.EnumerateFiles())
                {
                    file.Delete();
                }
                foreach (DirectoryInfo dir in di.EnumerateDirectories())
                {
                    dir.Delete(true);
                }
            }
            else
            {
                Directory.CreateDirectory(folder);
            }
        }

        public void save_ckpt(string checkpoint_name, string folder = "./model_ckpts")
        {
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }

            save(folder + "/" + checkpoint_name);
            Console.WriteLine($"Saved this model to " + folder + "/" + checkpoint_name);
        }

        public void load_ckpt(string checkpoint_name, string folder = "./model_ckpts")
        {
            if(Directory.Exists(folder))
            {
                load(folder + "/" + checkpoint_name);
            }
        }
    }

    public class main
    {
        static int Main()
        {
            torch.set_default_dtype(torch.float64);

            //KAN model = new KAN(width: new int[] { 2, 5, 1 }, grid: 5, k: 3, noise_scale: 0.1, seed: 0);
            //Func<Tensor, Tensor> f = (Tensor x) => torch.exp(torch.sin(np.pi * x[TensorIndex.Colon, TensorIndex.Slice(0, 1)]) + x[TensorIndex.Colon, TensorIndex.Slice(1, 2)].square());
            //Dictionary<string, Tensor> dataset = Utils.create_dataset(f, n_var: 2);
            //model.train(dataset, optim_name: "LBFGS", steps: 50, lambda: 0.01);

            //KAN model_coarse = new KAN(width: new int[] { 2, 5, 1 }, grid: 5, k: 3);
            //KAN model_fine = new KAN(width: new int[] { 2, 5, 1 }, grid: 10, k: 3);
            //Console.WriteLine(model_fine.act_funs[0].coef[0][0].ToDouble());
            //Tensor x = torch.normal(0, 1, size: new long[] { 100, 2 });
            //model_fine.initialize_grid_from_another_model(model_coarse, x);
            //Console.WriteLine(model_fine.act_funs[0].coef[0][0].ToDouble());

            //KAN model = new KAN(width: new int[] { 2, 5, 1 }, grid: 5, k: 3);
            //Console.WriteLine(model.act_funs[0].grid[0].ToString(style: TorchSharp.TensorStringStyle.Numpy));
            //Tensor x = torch.rand(new long[] { 100, 2 }) * 5;
            //model.update_grid_from_samples(x);
            //Console.WriteLine(model.act_funs[0].grid[0].ToString(style: TorchSharp.TensorStringStyle.Numpy));

            //KAN parent_model = new KAN(width : new int[] { 1,1}, grid: 5, k: 3);
            //parent_model.act_funs[0].grid.requires_grad_(false).set_(torch.linspace(-2, 2, steps: 6).unsqueeze(0)).requires_grad_(true);
            //Tensor x = torch.linspace(-2, 2, steps: 1001).unsqueeze(1);
            //KAN model = new KAN(width: new int[] { 1, 1 }, grid: 5, k: 3);
            //Console.WriteLine(model.act_funs[0].grid.ToString(style: TorchSharp.TensorStringStyle.Numpy));
            //model.initialize_grid_from_another_model(parent_model, x);
            //Console.WriteLine(model.act_funs[0].grid.ToString(style: TorchSharp.TensorStringStyle.Numpy));

            //KAN model = new KAN(width: new int[] { 2, 5, 3 }, grid: 5, k: 3);
            //Tensor x = torch.normal(0, 1, size: new long[] { 100, 2 });
            //Console.WriteLine(model.forward(x).ToString(style: TorchSharp.TensorStringStyle.Numpy));

            //KAN model = new KAN(width: new int[] { 2, 5, 1 }, grid: 5, k: 3, noise_scale: 1.0);
            //Tensor x = torch.normal(0, 1, size: new long[] { 100, 2 });
            //model.forward(x);
            //model.fix_symbolic(0, 1, 3, "sin", fit_parameters: true);
            //Console.WriteLine(model.act_funs[0].mask.reshape(2, 5).ToString(style: TorchSharp.TensorStringStyle.Numpy));
            //Console.WriteLine(model.symbolic_funs[0].mask.reshape(2, 5).ToString(style: TorchSharp.TensorStringStyle.Numpy));

            //KAN model = new KAN(width: new int[] { 2, 3, 1 }, grid: 5, k: 3, noise_scale: 1.0);
            //model.lock_(0, new NDArray(new long[][] { new long[] { 1, 0 }, new long[] { 1, 1 } }));
            //Console.WriteLine(model.act_funs[0].weight_sharing.reshape(3, 2).ToString(style: TorchSharp.TensorStringStyle.Numpy));
            //model.unlock_(0, new NDArray(new long[][] { new long[] { 1, 0 }, new long[] { 1, 1 } }));
            //Console.WriteLine(model.act_funs[0].weight_sharing.reshape(3, 2).ToString(style: TorchSharp.TensorStringStyle.Numpy));

            //KAN model = new KAN(width: new int[] { 2, 3, 1 }, grid: 5, k: 3, noise_scale: 1.0);
            //Tensor x = torch.normal(0, 1, size: new long[] { 100, 2 });
            //model.forward(x);
            //var res = model.get_range(0, 0, 0);
            //Console.WriteLine(res.Item1 + " " + res.Item2 + " " + res.Item3 + " " + res.Item4);

            int dim = 2, np_i = 21, np_b = 21;
            NDArray ranges = new NDArray(new double[] { 0, 2 * Math.PI });

            KAN model = new KAN(width: new int[] { 1, 1, 1 }, grid: 5, k: 3, grid_eps: 1.0, noise_scale_base: 0.25);

            //Func<Tensor, Tensor> sol_fun = (Tensor x) => torch.sin(np.pi * x[TensorIndex.Colon, TensorIndex.Slice(0, 1)]) *
            //                                                torch.sin(np.pi * x[TensorIndex.Colon, TensorIndex.Slice(1, 2)]);
            //Func<Tensor, Tensor> source_fun = (Tensor x) => -2 * np.pi * np.pi * torch.sin(np.pi * x[TensorIndex.Colon, TensorIndex.Slice(0, 1)]) *
            //                                                                        torch.sin(np.pi * x[TensorIndex.Colon, TensorIndex.Slice(1, 2)]);

            Func<Tensor, Tensor> sol_fun = (Tensor x) => torch.sin(Math.PI * x) + 1;
            Func<Tensor, Tensor> source_fun = (Tensor x) => -Math.PI * Math.PI * torch.sin(Math.PI * x);

            string sampling_mode = "random";
            Tensor x_i = torch.ones(1), 
                    x_mesh = torch.linspace(ranges[0], ranges[1], steps: np_i, dtype: torch.float64), 
                    y_mesh = torch.linspace(ranges[0], ranges[1], steps: np_i, dtype: torch.float64);
            Tensor[] X_Y = torch.meshgrid(new Tensor[] { x_mesh, y_mesh });

            switch (sampling_mode)
            {
                case "mesh":
                    x_i = torch.stack(new Tensor[] { X_Y[0].reshape(-1), X_Y[1].reshape(-1) }).permute(1, 0);
                    break;
                case "random":
                    //x_i = torch.rand(new long[] { np_i * np_i, 2 }, dtype: torch.float64) * 2 - 1;
                    x_i = torch.rand(new long[] { np_i * np_i, 1 }, dtype: torch.float64) * 2 * Math.PI - 2 * Math.PI;
                    break;
            }

            x_i = x_i.requires_grad_(true);

            //Func<Tensor, Tensor, Tensor> helper = (Tensor X, Tensor Y) => torch.stack(new Tensor[] { X.reshape(-1), Y.reshape(-1) }).permute(1, 0);
            //Tensor xb1 = helper(X_Y[0][0], X_Y[1][0]);
            //Tensor xb2 = helper(X_Y[0][-1], X_Y[1][0]);
            //Tensor xb3 = helper(X_Y[0][TensorIndex.Colon, 0], X_Y[1][TensorIndex.Colon, 0]);
            //Tensor xb4 = helper(X_Y[0][TensorIndex.Colon, 0], X_Y[1][TensorIndex.Colon, -1]);
            //Tensor x_b = torch.cat(new Tensor[] { xb1, xb2, xb3, xb4 }, dim: 0).requires_grad_(true);

            Tensor x_b = torch.cat(new Tensor[] { torch.full(new long[] { 1, 1 }, 0), 
                                                    torch.ones(new long[] { 1, 1 }), 
                                                    torch.full(new long[] { 1, 1 }, 2) }, dim: 0).requires_grad_(true);

            int steps = 10, log = 1;
            double alpha = 0.1;

            void train()
            {
                optim.Optimizer optimizer = LBFGS(model.parameters(), lr: 1e-1, history_size: 10, tolerange_grad: 1e-32, tolerance_change: 1e-32);

                Tensor pde_loss = torch.zeros(1), bc_loss = torch.zeros(1);
                Func<Tensor> closure = () =>
                {
                    optimizer.zero_grad();

                    Tensor pred = model.forward(x_i);
                    Tensor sol_D1 = autograd.grad(new Tensor[] { pred }, new Tensor[] { x_i },
                                                    new Tensor[] { torch.ones_like(pred) }, create_graph: true, retain_graph: true)[0];
                    Tensor sol_D2 = autograd.grad(new Tensor[] { sol_D1 }, new Tensor[] { x_i },
                                                    new Tensor[] { torch.ones_like(sol_D1) }, create_graph: true, retain_graph: true)[0];
                    Tensor source = source_fun(x_i);
                    pde_loss = torch.mean(torch.square(sol_D2 - source), type: torch.float64, dimensions: new long[] { 0 });
                    
                    Tensor bc_true = sol_fun(x_b);
                    Tensor bc_pred = model.forward(x_b);
                    bc_loss = torch.mean(torch.square(bc_pred - bc_true), type: torch.float64, dimensions: new long[] { 0 });

                    Tensor loss = alpha * pde_loss + bc_loss;
                    loss.backward(create_graph: true);
                    return loss;
                };

                using (ProgressBar pbar = new ProgressBar((int)Math.Ceiling((double)steps / log), "Starting training...", new ProgressBarOptions { ProgressBarOnBottom = true }))
                {
                    for (int i = 0; i < steps; ++i)
                    {
                        if (i % 5 == 0 && i < 50)
                        {
                            model.update_grid_from_samples(x_i);
                        }

                        optimizer.step(closure);
                        Tensor sol = sol_fun(x_i);
                        Tensor l2 = torch.mean(torch.square(model.forward(x_i) - sol), type: torch.float64, dimensions: new long[] { 0 });

                        if (i % log == 0)
                        {
                            //Console.WriteLine($"PDE loss: {pde_loss.detach().cpu().ToDouble():e2} | BC loss: {bc_loss.detach().cpu().ToDouble():e2} | l2: {l2.detach().cpu().ToDouble():e2}");
                            pbar.Tick($"PDE loss: {pde_loss.detach().cpu().ToDouble():e2} | BC loss: {bc_loss.detach().cpu().ToDouble():e2} | l2: {l2.detach().cpu().ToDouble():e2}");
                        }
                    }
                }
            }

            train();

            //model.suggest_symbolic(0, 0, 0);
            //model.suggest_symbolic(1, 0, 0);

            //model.fix_symbolic(0, 0, 0, "x");
            //model.fix_symbolic(1, 0, 0, "sin");

            //for (int i = 0; i < 2; ++i)
            //{
            //    for (int j = 0; j < 2; ++j)
            //    {
            //        model.fix_symbolic(0, i, j, "x");
            //    }

            //    model.fix_symbolic(1, i, 0, "sin");
            //}

            train();

            return 0;
        }
    }
}
