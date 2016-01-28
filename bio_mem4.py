import time
import numpy as np
import scipy.optimize
import scipy.stats

import nengo
from nengo.networks import EnsembleArray
import ctn_benchmark
from ctn_benchmark.stats import bootstrapci


class BioSpaunMemory(ctn_benchmark.Benchmark):
    def params(self):
        self.default('number of neurons', n_neurons=4000)
        self.default('number of dimensions', D=2)
        self.default('maximum firing rate', max_rate=80)
        self.default('stimulus strength', stim_mag=4)
        self.default('amount of neuron noise', neuron_noise=0.005)
        self.default('additive bias', neuron_bias=0.0)  # 0.0002, -0.0006
        self.default('recurrent synapse', synapse_memory=0.1)
        self.default('memory feedback decay value', memory_decay=1.0)
        self.default('do curve fitting', do_fit=False)
        self.default('number of trials to average over', n_trials=1)
        self.default('empirical dataset', dataset='pre_PHE')
        self.default('plot type', plot_type='all')
        self.default('noise of memory estimation', noise_readout=0.35)
        self.default('misperception prob', misperceive=0.07)
        self.default('simulation time', simtime=10.0)
        self.default('ramp input scale', ramp_scale=0.1)

    def model(self, p):
        model = nengo.Network()
        model.config[nengo.Ensemble].max_rates = \
            nengo.dists.Uniform(p.max_rate / 2, p.max_rate)

        with model:
            stim = nengo.Node(lambda t: 1 if 0 < t < 1 else 0)
            # ramp = nengo.Node(lambda t: 0 if t < 1 else (t - 1.0) / p.simtime)
            ramp = nengo.Node(lambda t: t > 1)

            sensory = EnsembleArray(n_neurons=100, n_ensembles=p.D)

            memory = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D)
            memory.noise = \
                nengo.processes.WhiteNoise(
                    dist=nengo.dists.Gaussian(mean=p.neuron_bias,
                                              std=p.neuron_noise))

            nengo.Connection(stim, sensory.input[0], synapse=None)
            nengo.Connection(ramp, sensory.input[1], synapse=None)
            nengo.Connection(sensory.output[0], memory[0],
                             synapse=p.synapse_memory,
                             transform=p.stim_mag * p.synapse_memory)
            nengo.Connection(sensory.output[1], memory[1],
                             synapse=p.synapse_memory,
                             transform=p.synapse_memory * p.ramp_scale)

            nengo.Connection(memory, memory, synapse=p.synapse_memory,
                             transform=p.memory_decay)

            self.p_mem = nengo.Probe(memory, synapse=0.1, sample_every=0.5)
            self.p_mem2 = nengo.Probe(memory, synapse=0.1, sample_every=0.1)
            self.p_r = nengo.Probe(ramp)
        return model

    def get_exp_data(self, dataset):
        exp_data_dict = {
            'pre_PHE': [0.972, 0.947, 0.913, 0.798],  # experimental data from WebPlotDigitizer  # noqa
            'post_PHE': [0.972, 0.938, 0.847, 0.666],  # 800-1200 trials
            'pre_GFC': [0.970, 0.942, 0.882, 0.766],
            'post_GFC': [0.966, 0.928, 0.906, 0.838]}

        return exp_data_dict[dataset]

    def evaluate(self, p, sim, plt):
        model_results = []
        integrator_values = []
        exp_data = self.get_exp_data(p.dataset)

        if p.seed == 0:
            p.seed = int(time.time())

        def curve(x, noise, ignore):
            return (scipy.stats.norm.cdf(x / noise) * (1 - p.misperceive)
                    + 0.5 * p.misperceive)

        for trial in range(p.n_trials):
            model = self.model(p)
            model.seed = p.seed + trial

            sim = nengo.Simulator(model)
            sim.run(p.simtime)
            self.record_speed(p.simtime * p.n_trials)

            values = sim.data[self.p_mem][[5, 9, 13, 17], 0]   # model data

            model_results.append(values)
            integrator_values.append(sim.data[self.p_mem2][:, 0])

        print "DONE RUNNING MODELS"

        mean_model_results = np.mean(model_results, axis=0)
        ci_model_results = np.array([bootstrapci(d, np.mean)
                                     for d in np.array(model_results).T])

        if p.do_fit:
            cp, err = scipy.optimize.curve_fit(curve, mean_model_results,
                                               exp_data)
        else:
            cp = [p.noise_readout, p.misperceive]

        curve_results = curve(mean_model_results, *cp)
        ci_0_model_results = curve(ci_model_results[:, 0], *cp)
        ci_1_model_results = curve(ci_model_results[:, 1], *cp)

        print "DONE PROCESSING MODEL RESULTS"

        mean_integrator_values = np.mean(integrator_values, axis=0)
        ci_integrator_values = \
            np.array([bootstrapci(d, np.mean)
                      for d in np.array(integrator_values).T])

        print "DONE PROCESSING INT VALUES"

        if plt is not None:
            plt.subplot(3, 1, 1)
            plt.fill_between([2, 4, 6, 8], ci_0_model_results,
                             ci_1_model_results, color='#aaaaaa')
            plt.plot([2, 4, 6, 8], curve_results,
                     label='model ($\sigma$=%0.2f, ig=%1.3f)' % (cp[0], cp[1]))
            plt.plot([2, 4, 6, 8], exp_data, label='exp data')
            plt.legend(loc='best')

            plt.subplot(3, 1, 2)
            plt.fill_between(sim.trange(self.p_mem2.sample_every),
                             ci_integrator_values[:, 0],
                             ci_integrator_values[:, 1], color='#aaaaaa')
            plt.plot(sim.trange(self.p_mem2.sample_every),
                     mean_integrator_values)

            plt.subplot(3, 1, 3)
            plt.plot(sim.trange(self.p_mem2.sample_every),
                     sim.data[self.p_mem2])
            plt.plot(sim.trange(), sim.data[self.p_r])

        print "DONE PLOTTING"

        return dict(rmse=np.sqrt(np.mean(curve_results - exp_data) ** 2),
                    choice_noise=cp[0],
                    ignore=cp[1],
                    values=values.tolist())

if __name__ == '__main__':
    BioSpaunMemory().run()
