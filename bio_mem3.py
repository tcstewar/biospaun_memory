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
        self.default('additive bias', neuron_bias=0.0)
        self.default('recurrent synapse', synapse_memory=0.1)
        self.default('do curve fitting', do_fit=False)
        #self.default('memory time', tau_memory=2.0)

    def model(self, p):
        model = nengo.Network()
        model.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(p.max_rate/2, 
                                                                     p.max_rate)
        with model:
            stim = nengo.Node(lambda t: 1 if 0<t<1 else 0)

            sensory = nengo.Ensemble(n_neurons=100, dimensions=1)

            memory = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D)
            memory.noise = nengo.processes.WhiteNoise(
                                            dist=nengo.dists.Gaussian(mean=p.neuron_bias, 
                                                   std=p.neuron_noise))

            nengo.Connection(stim, sensory, synapse=None)
            nengo.Connection(sensory, memory[0], synapse=p.synapse_memory, transform=p.stim_mag*p.synapse_memory)

            #transform = 1.0 - p.synapse_memory / p.tau_memory
            transform = 1.0
        
            def mem_func(x):
                return x
                if np.linalg.norm(x)<0.2:
                    return x*0
                else:
                    return x
            nengo.Connection(memory, memory, synapse=p.synapse_memory, function=mem_func, transform=transform)

            self.p_mem = nengo.Probe(memory, synapse=0.1, sample_every=0.5)
            self.p_mem2 = nengo.Probe(memory, synapse=0.1)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(9)
        self.record_speed(9)

        values = sim.data[self.p_mem][[5,9,13,17],0]   # model data
        data = [0.97, 0.94, 0.91, 0.80]                # experimental data

        def curve(x, noise, ignore):
            return scipy.stats.norm.cdf(values/noise)*(1-ignore) + 0.5*ignore

        if p.do_fit:
            p, err = scipy.optimize.curve_fit(curve, np.arange(3), data)
        else:
            p = [0.35, 0.07]
        
        rmse = np.sqrt(np.mean((curve(0, *p)-data)**2))

        if plt is not None:
            plt.subplot(2,1,1)
            plt.plot([2,4,6,8], curve(0, *p), label='model ($\sigma$=%0.2f, ig=%1.3f)' % (p[0], p[1]))
            plt.plot([2,4,6,8], data, label='exp')
            plt.legend(loc='best')
            plt.subplot(2,1,2)
            plt.plot(sim.trange(), sim.data[self.p_mem2][:,0])
        
        return dict(rmse=rmse, 
                    choice_noise=p[0],
                    ignore=p[1],
                    values=values.tolist())

if __name__ == '__main__':
    BioSpaunMemory().run()
