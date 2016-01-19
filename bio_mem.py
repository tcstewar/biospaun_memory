import ctn_benchmark
import nengo
import scipy.optimize
import scipy.stats
import numpy as np

class BioSpaunMemory(ctn_benchmark.Benchmark):
    def params(self):
        self.default('number of neurons', n_neurons=100)
        self.default('number of dimensions', D=8)
        self.default('maximum firing rate', max_rate=80)
        self.default('stimulus strength', stim_mag=1)
        self.default('amount of neuron noise', neuron_noise=0.01)
        self.default('recurrent synapse', synapse_memory=0.1)

    def model(self, p):
        model = nengo.Network()
        model.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(p.max_rate/2, 
                                                                     p.max_rate)
        with model:
            stim = nengo.Node(lambda t: p.stim_mag if 0<t<1 else 0)

            sensory = nengo.Ensemble(n_neurons=100, dimensions=1)

            memory = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D)
            memory.noise = nengo.processes.WhiteNoise(
                                            dist=nengo.dists.Gaussian(mean=0, 
                                                   std=p.neuron_noise))

            nengo.Connection(stim, sensory, synapse=None)
            nengo.Connection(sensory, memory[0], synapse=0.01)
            nengo.Connection(memory, memory, synapse=p.synapse_memory)

            self.p_mem = nengo.Probe(memory, synapse=0.1, sample_every=0.5)
        return model

    def evaluate_model(self, p, Simulator, model, plt):
        averages=20     #how to input this as a parameter?
        model_data=[]
        rng = np.random.RandomState(seed=p.seed)
        for a in range(averages):
            model.seed=rng.randint(0x7FFFFFFF)
            sim = Simulator(model, seed=rng.randint(0x7FFFFFFF))
            sim.run(9)
            model_data.append(sim.data[self.p_mem][[5,9,13,17],0])
        self.record_speed(9*averages)

        mean_values = np.mean(model_data, axis=0)   # model data
        std_values = np.std(model_data, axis=0)
        exp_values_pre_PHE = [0.972, 0.947, 0.913, 0.798]  # experimental data from WebPlotDigitizer
        exp_values_post_PHE = [0.972, 0.938, 0.847, 0.666]  # 800-1200 trials
        exp_values_pre_GFC = [0.970, 0.942, 0.882, 0.766]
        exp_values_post_GFC = [0.966, 0.928, 0.906, 0.838]
        

        ci = np.array([ctn_benchmark.stats.bootstrapci(d, np.mean) for d in np.array(model_data).T])

        def curve(x, noise):
            return scipy.stats.norm.cdf(x/noise)

        p, err = scipy.optimize.curve_fit(curve, mean_values, exp_values_pre_GFC)
        
        rmse = np.sqrt(np.mean((curve(mean_values, *p)-exp_values_pre_GFC)**2))

        if plt is not None:
            plt.fill_between([2,4,6,8], curve(ci[:,0], *p), curve(ci[:,1], *p), color='#aaaaaa')
            plt.plot([2,4,6,8], curve(mean_values, *p), label='model_data ($\sigma$=%0.2f)' % p[0])
            plt.plot([2,4,6,8], exp_values_pre_GFC, label='exp_values_pre_GFC')
            #plt.plot([2,4,6,8], curve(ci[:,0], *p), lw=3)
            #plt.plot([2,4,6,8], curve(ci[:,1], *p), lw=3)

            #for d in model_data:
            #    plt.plot([2,4,6,8], curve(d, *p))


            plt.legend(loc='best')
        
        return dict(rmse=rmse, 
                    choice_noise=p[0],)

if __name__ == '__main__':
    BioSpaunMemory().run()
