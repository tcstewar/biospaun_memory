import bio_mem
import ctn_benchmark
import numpy as np

def task_run_neurons():
    def run():
        for seed in range(10, 30):
            for n_neurons in [20, 50, 100, 200, 500, 1000]:
                bio_mem.BioSpaunMemory().run(seed=seed,
                                             n_neurons=n_neurons,
                                             D=8,
                                             neuron_noise=0.01)
    return dict(actions=[run], verbosity=2)

def task_plot_neurons():
    def plot():
        data = ctn_benchmark.Data('data')

        plt = ctn_benchmark.Plot(data)
        plt.lines('_n_neurons', ['rmse'])

        import pylab
        pylab.show()
    return dict(actions=[plot], verbosity=2)

def task_fit():
    def fit():
        data = ctn_benchmark.Data('data')
        n_neurons = 200
        for d in data.data[:]:
            if d['_n_neurons'] != n_neurons:
                data.data.remove(d)

        target_data = [0.97, 0.94, 0.91, 0.80]           # experimental data

        import scipy.stats
        def prob(x, noise=1.0):
            return scipy.stats.norm.cdf(np.array(x)/noise)
        def curve(x, noise):
            return np.mean([prob(d['values'], noise=noise) for d in data.data], axis=0)

        import scipy.optimize
        p, err = scipy.optimize.curve_fit(curve, np.arange(3), target_data)

        import pylab
        pylab.plot([2,4,6,8], target_data)
        pylab.plot([2,4,6,8], curve(None, *p))
        pylab.show()

        print p, err
    return dict(actions=[fit], verbosity=2)
