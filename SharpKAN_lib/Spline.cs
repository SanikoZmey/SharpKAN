using TorchSharp;
using static TorchSharp.torch;

namespace SharpKAN_lib
{
    public class Spline
    {
        static Tensor B_batch(Tensor x, Tensor grid, int k = 0, bool extend = true, string device = "cpu")
        {
            Tensor extend_Grid(Tensor grid_to_ext, int k_extend = 0)
            {
                Tensor h = (grid_to_ext[TensorIndex.Colon, TensorIndex.Slice(-1, null)] - grid_to_ext[TensorIndex.Colon, TensorIndex.Slice(0, 1)]) / (grid_to_ext.shape[1] - 1);
                
                for (int i = 0; i < k_extend; ++i)
                {
                    grid_to_ext = torch.cat(new Tensor[] { grid_to_ext[TensorIndex.Colon, TensorIndex.Slice(0, 1)] - h, grid_to_ext }, dim: 1);
                    grid_to_ext = torch.cat(new Tensor[] { grid_to_ext, grid_to_ext[TensorIndex.Colon, TensorIndex.Slice(-1, null)] + h }, dim: 1);
                }

                grid_to_ext = grid_to_ext.to(device);
                return grid_to_ext;
            }

            if (extend)
            {
                grid = extend_Grid(grid, k);
            }

            grid = grid.unsqueeze(2).to(device);
            x = x.unsqueeze(1).to(device);

            Tensor value = (x >= grid[TensorIndex.Colon, TensorIndex.Slice(null, -1)]) * (x < grid[TensorIndex.Colon, TensorIndex.Slice(1, null)]);

            for (int i = 1; i < k + 1; ++i)
            {
                value = (x - grid[TensorIndex.Colon, TensorIndex.Slice(null, -(i + 1))]) / (grid[TensorIndex.Colon, TensorIndex.Slice(i, -1)] - grid[TensorIndex.Colon, TensorIndex.Slice(null, -(i + 1))]) * value[TensorIndex.Colon, TensorIndex.Slice(null, -1)] +
                        (grid[TensorIndex.Colon, TensorIndex.Slice(i + 1, null)] - x) / (grid[TensorIndex.Colon, TensorIndex.Slice(i + 1, null)] - grid[TensorIndex.Colon, TensorIndex.Slice(1, -i)]) * value[TensorIndex.Colon, TensorIndex.Slice(1, null)];
            }

            return value;
        }

        public static Tensor coeff2curve(Tensor x_eval, Tensor grid, Tensor coeff, int k, string device = "cpu")
        {
            if (coeff.dtype != x_eval.dtype)
            {
                coeff = coeff.to_type(x_eval.dtype);
            }

            return torch.einsum("ij,ijk->ik", coeff, B_batch(x_eval, grid, k, device: device));
        }

        public static Tensor curve2coeff(Tensor x_eval, Tensor y_eval, Tensor grid, int k, string device = "cpu")
        {
            Tensor mat = B_batch(x_eval, grid, k, device: device).permute(0, 2, 1);

            return torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(2).to(device)).Solution[TensorIndex.Colon, TensorIndex.Colon, 0];
        }

        //static int Main()
        //{
        //    torch.set_default_dtype(torch.float64);

        //    int num_splines = 5, num_samples = 100, num_grid_intervals = 10, k = 3;
        //    Tensor x_eval = torch.normal(0, 1, size: new long[] { num_splines, num_samples });
        //    Tensor y_eval = torch.normal(0, 1, size: new long[] { num_splines, num_samples });
        //    Tensor grids = torch.einsum("i,j->ij", torch.ones(num_splines), torch.linspace(-1, 1, steps: num_grid_intervals + 1));
        //    Console.WriteLine(curve2coeff(x_eval, y_eval, grid: grids, k: k));


        //    //Func<Tensor, Tensor> f = (Tensor x) => torch.exp(torch.sin(np.pi * x[TensorIndex.Colon, TensorIndex.Slice(0, 1)]) +
        //    //                                        torch.square(x[TensorIndex.Colon, TensorIndex.Slice(1, 2)]));

        //    //Dictionary<string, Tensor> dataset = create_dataset(f, n_var: 2, train_num: 100);
        //    //Console.WriteLine(dataset["train_input"].ToString(style: TorchSharp.TensorStringStyle.Numpy));
        //    return 0;
        //}
    }
}
