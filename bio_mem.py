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

    def evaluate(self, p, sim, plt):
        sim.run(9)
        self.record_speed(9)

        values = sim.data[self.p_mem][[5,9,13,17],0]   # model data
        data = [0.97, 0.94, 0.91, 0.80]                # experimental data

        def curve(x, noise):
            return scipy.stats.norm.cdf(values/noise)

        p, err = scipy.optimize.curve_fit(curve, np.arange(3), data)
        
        rmse = np.sqrt(np.mean((curve(0, *p)-data)**2))

        if plt is not None:
            plt.plot([2,4,6,8], curve(0, *p), label='model ($\sigma$=%0.2f)' % p[0])
            plt.plot([2,4,6,8], data, label='exp')
            plt.legend(loc='best')
        
        return dict(rmse=rmse, 
                    choice_noise=p[0],
                    values=values.tolist())

if __name__ == '__main__':
    BioSpaunMemory().run()
