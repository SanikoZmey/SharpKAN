using System;
using NumSharp;
using TorchSharp;
using static TorchSharp.torch;
using static SharpKAN_lib.Spline;
using TorchSharp.Modules;


namespace SharpKAN_lib
{
    public class KANLayer : nn.Module<Tensor, Tuple<Tensor, Tensor, Tensor, Tensor>>
    {
        public int in_dim, out_dim, size, k;
        private long num, lock_counter;
        private double grid_eps;
        public bool sp_trainable, sb_trainable;
        string device;

        public Tensor weight_sharing, lock_id;
        public Parameter grid, coef, scale_base, scale_sp, mask;
        private torch.nn.Module<torch.Tensor, torch.Tensor> base_fun;
        public KANLayer(int in_dim = 3, int out_dim = 2, long num = 5, int k = 3, double noise_scale = 0.1, Tensor scale_base = null, double scale_sp = 1,
                        torch.nn.Module<torch.Tensor, torch.Tensor> base_fun = null, double grid_eps = 0.02, NDArray grid_range = null,
                        bool sp_trainable = true, bool sb_trainable = true, string device = "cpu") : base(nameof(KANLayer))
        {
            torch.set_default_dtype(torch.float64);

            if (base_fun == null)
            {
                base_fun = nn.SiLU();
            }

            if(grid_range == null)
            {
                grid_range = new NDArray(new double[] { -1.0, 1.0});
            }

            if(scale_base is null)
            {
                scale_base = torch.ones(new long[] { 1 });
            }

            this.in_dim = in_dim;
            this.out_dim = out_dim;
            this.size = this.in_dim * this.out_dim;
            this.num = num;
            this.k = k;
            this.sb_trainable = sb_trainable;
            this.sp_trainable = sp_trainable;

            this.grid = nn.Parameter(torch.einsum("i,j->ij", torch.ones(this.size, device: device), torch.linspace(grid_range[0], grid_range[1], steps: this.num + 1, device: device)), 
                                    requires_grad: false);
            using (Tensor noises = (torch.rand(this.size, this.grid.shape[1], dtype: torch.float64, device: device) - 1 / 2) * noise_scale / num)
                this.coef = nn.Parameter(curve2coeff(this.grid, noises, this.grid, k, device));
            this.scale_base = nn.Parameter(torch.ones(this.size, device: device) * scale_base, requires_grad: this.sb_trainable);
            this.scale_sp = nn.Parameter(torch.ones(this.size, device: device) * scale_sp, requires_grad: this.sp_trainable);
            this.base_fun = base_fun;

            this.mask = nn.Parameter(torch.ones(this.size, device: device), requires_grad: false);
            this.grid_eps = grid_eps;
            this.weight_sharing = torch.arange(this.size);
            this.lock_counter = 0;
            this.lock_id = torch.zeros(this.size);
            this.device = device;

            RegisterComponents();
        }

        public override Tuple<Tensor, Tensor, Tensor, Tensor> forward(Tensor x)
        {
            long batch_size = x.shape[0];
            x = torch.einsum("ij,k->ikj", x, torch.ones(out_dim, device: device));

            Tensor y, postacts, postspline;
            using (Tensor x_reshaped = x.reshape(new long[] { batch_size, size }).permute(new long[] { 1, 0 }),
                          base_ = base_fun.forward(x_reshaped).permute(new long[] { 1, 0 }))
            {
                y = coeff2curve(x_reshaped, grid[weight_sharing], coef[weight_sharing], k, device).permute(new long[] { 1, 0 });
                postspline = y.clone().reshape(new long[] { batch_size, out_dim, in_dim });
                y = (scale_base.unsqueeze(0) * base_ + scale_sp.unsqueeze(0) * y) * mask[TensorIndex.None, TensorIndex.Colon];
            }
            y = y.reshape(new long[] { batch_size, out_dim, in_dim });
            postacts = y.clone();
            y = torch.sum(y, type: torch.float64, dim: 2);

            return new Tuple<Tensor, Tensor, Tensor, Tensor>(y, x.clone(), postacts, postspline);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
        }

        public void update_grid_from_samples(Tensor x)
        {
            torch.set_default_dtype(torch.float64);

            long batch_size = x.shape[0];
            x = torch.einsum("ij,k->ikj", x, torch.ones(out_dim, device: device)).reshape(new long[] { batch_size, size }).permute(new long[] { 1, 0 });
            Tensor x_pos = torch.sort(x, dim: 1).values;
            Tensor y_eval = coeff2curve(x_pos, grid, coef, k, device);
            long num_intervals = grid.shape[1] - 1;
            long[] ids = new long[num_intervals + 1];
            for(int i = 0; i < num_intervals; ++i)
            {
                ids[i] = (batch_size / num_intervals * i);
            }
            ids[num_intervals] = x_pos.shape[1] - 1;

            Tensor grid_adaptive = x_pos[TensorIndex.Colon, TensorIndex.Tensor(ids)];
            double margin = 0.01;
            Tensor[] grid_to_cat = new Tensor[grid.shape[1]];
            int counter = 0;
            foreach(double a in torch.linspace(0, 1, steps: grid.shape[1], dtype: torch.float64).data<double>())
            {
                grid_to_cat[counter++] = grid_adaptive[TensorIndex.Colon, TensorIndex.Slice(0, 1)] - margin +
                                        (grid_adaptive[TensorIndex.Colon, TensorIndex.Slice(-1, null)] - grid_adaptive[TensorIndex.Colon, TensorIndex.Slice(0, 1)] + 2 * margin) * a;
            }
            Tensor grid_uniform = torch.cat(grid_to_cat, dim: 1);
            grid.set_(grid_eps * grid_uniform.clone() + (1 - grid_eps) * grid_adaptive);
            coef.requires_grad_(false).set_(curve2coeff(x_pos, y_eval, grid, k, device)).requires_grad_(true);
        }

        public void initialize_grid_from_parent(KANLayer parent, Tensor x)
        {
            torch.set_default_dtype(torch.float64);

            long batch_size = x.shape[0];
            Tensor x_eval = torch.einsum("ij,k->ikj", x, torch.ones(out_dim, device: device)).reshape(batch_size, size).permute(1, 0);
            Tensor x_pos = parent.grid;
            KANLayer sp2 = new KANLayer(in_dim: 1, out_dim: size, k: 1, num : x_pos.shape[1] - 1, scale_base : 0, device: device);
            sp2.coef.requires_grad_(false).set_(curve2coeff(sp2.grid, x_pos, sp2.grid, k:1, device: device)).requires_grad_(true);
            Tensor y_eval = coeff2curve(x_eval, parent.grid, parent.coef, k: parent.k, device: device);
            Tensor percentile = torch.linspace(-1, 1, steps: num + 1, dtype: torch.float64, device: device);
            grid.set_(sp2.forward(percentile.unsqueeze(1)).Item1.permute(1, 0));
            coef.requires_grad_(false).set_(curve2coeff(x_eval, y_eval, grid, k, device)).requires_grad_(true);
        }

        public KANLayer get_subset(long[] in_id,  long[] out_id)
        {
            torch.set_default_dtype(torch.float64);

            KANLayer spb = new KANLayer(in_id.Length, out_id.Length, num, k, base_fun: base_fun, device: device);
            spb.grid.set_(grid.reshape(new long[] { out_dim, in_dim, spb.num + 1 })[TensorIndex.Tensor(out_id)][TensorIndex.Colon, TensorIndex.Tensor(in_id)].reshape(new long[] { -1, spb.num + 1 }));
            spb.coef.requires_grad_(false).set_(coef.reshape(new long[] { out_dim, in_dim, spb.coef.shape[1] })[TensorIndex.Tensor(out_id)][TensorIndex.Colon, TensorIndex.Tensor(in_id)].reshape(new long[] { -1, spb.coef.shape[1] })).requires_grad_(true);
            spb.scale_base.requires_grad_(false).set_(scale_base.reshape(new long[] { out_dim, in_dim })[TensorIndex.Tensor(out_id)][TensorIndex.Colon, TensorIndex.Tensor(in_id)].reshape(new long[] { -1 })).requires_grad_(sb_trainable);
            spb.scale_sp.requires_grad_(false).set_(scale_sp.reshape(new long[] { out_dim, in_dim })[TensorIndex.Tensor(out_id)][TensorIndex.Colon, TensorIndex.Tensor(in_id)].reshape(new long[] { -1 })).requires_grad_(sp_trainable);
            spb.mask.set_(mask.reshape(new long[] { out_dim, in_dim })[TensorIndex.Tensor(out_id)][TensorIndex.Colon, TensorIndex.Tensor(in_id)].reshape(new long[] { -1 }));
            
            return spb;
        }

        public void lock_(NDArray ids)
        {
            lock_counter++;
            for(int i = 0; i < ids.shape[0]; ++i)
            {
                if(i != 0)
                {
                    weight_sharing[ids[i][1] * in_dim + ids[i][0]] = (long)(ids[0][1] * in_dim + ids[0][0]);
                }
                lock_id[ids[i][1] * in_dim + ids[i][0]] = lock_counter;
            }
        }

        public void unlock_(NDArray ids)
        {
            bool locked = true;
            for(int i = 0; i < ids.shape[0]; ++i)
            {
                locked &= ((weight_sharing[ids[i][1] * in_dim + ids[i][0]]) == (weight_sharing[ids[0][1] * in_dim + ids[0][0]])).ToBoolean();
            }
            if(!locked)
            {
                Console.WriteLine("Given activation fuunctions are not locked. unlock failed.");
            }
            else
            {
                for(int i = 0; i < ids.shape[0]; ++i)
                {
                    weight_sharing[ids[i][1] * in_dim + ids[i][0]] = (long)(ids[i][1] * in_dim + ids[i][0]);
                    lock_id[ids[i][1] * in_dim + ids[i][0]] = 0;
                }
                lock_counter--;
            }
        }
    }
    //public class main
    //{
    //    static int Main()
    //    {
    //        torch.set_default_dtype(torch.float64);

    //        //KANLayer kan_large = new KANLayer(in_dim: 10, out_dim: 10, num: 5, k: 3);
    //        //KANLayer kan_small = kan_large.get_subset(new long[] { 0, 9 }, new long[] { 1, 2, 3 });
    //        //Console.WriteLine(kan_small.in_dim + " " + kan_small.out_dim);

    //        //KANLayer model = new KANLayer(in_dim: 1, out_dim: 1);

    //        //Tensor x = torch.normal(0, 1, size: new long[] { 100, 3 });
    //        //Tuple<Tensor, Tensor, Tensor, Tensor> pred = model.forward(x);
    //        ////Tensor loss = torch.mean(torch.square(pred.Item1 - torch.ones_like(x)), dimensions: new long[] { 0 });
    //        //Console.WriteLine(pred.Item1.ToString(style: TorchSharp.TensorStringStyle.Numpy) + "\n" + 
    //        //                    pred.Item2.ToString(style: TorchSharp.TensorStringStyle.Numpy) + "\n" + 
    //        //                    pred.Item3.ToString(style: TorchSharp.TensorStringStyle.Numpy) + "\n" + 
    //        //                    pred.Item4.ToString(style: TorchSharp.TensorStringStyle.Numpy));

    //        //int batch_size = 100;
    //        //KANLayer parent_model = new KANLayer(in_dim: 1, out_dim: 1);
    //        //Console.WriteLine(parent_model.grid.ToString(style: TorchSharp.TensorStringStyle.Numpy));
    //        ////Tensor x = torch.linspace(-3, 3, steps: 100).unsqueeze(1);
    //        ////parent_model.update_grid_from_samples(x);
    //        //KANLayer model = new KANLayer(in_dim: 1, out_dim: 1, num: 10);
    //        //Tensor x = torch.normal(0, 1, size: new long[] { batch_size, 1 });
    //        //model.initialize_grid_from_parent(parent_model, x);
    //        //Console.WriteLine(model.grid.ToString(style: TorchSharp.TensorStringStyle.Numpy));

    //        //KANLayer model = new KANLayer(in_dim: 2, out_dim: 2);
    //        //int counter = 0;
    //        //foreach (Parameter parameter in model.parameters())
    //        //{
    //        //    Console.WriteLine(parameter + " " + counter++);
    //        //}
    //        //model.lock_(new NDArray(new long[][] { new long[] { 0, 0 }, new long[] { 1, 2 }, new long[] { 2, 1 } }));
    //        //Console.WriteLine(model.weight_sharing.reshape(3, 3).ToString(style: TorchSharp.TensorStringStyle.Numpy));
    //        //model.unlock_(new NDArray(new long[][] { new long[] { 0, 0 }, new long[] { 1, 2 }, new long[] { 2, 1 } }));
    //        //Console.WriteLine(model.weight_sharing.reshape(3, 3).ToString(style: TorchSharp.TensorStringStyle.Numpy));

    //        return 0;
    //    }
    //}
}
