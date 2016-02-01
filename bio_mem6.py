import time
import numpy as np
import scipy.optimize
import scipy.stats

import nengo
from nengo.networks import EnsembleArray
import ctn_benchmark
from ctn_benchmark.stats import bootstrapci


class CurrentNode(nengo.Node):
    def __init__(self, encoders, gain=1.0, bias=0.0):
        self.encoders = encoders
        self.gain = gain
        self.bias = bias
        super(CurrentNode, self).__init__(self.step, size_in=encoders.shape[1],
                                          size_out=encoders.shape[0])

    def set_encoders(self, new_encoders):
        self.encoders[:] = new_encoders[:]

    def step(self, t, x):
        return np.dot(self.encoders, x) * self.gain + self.bias


class BioSpaunMemory(ctn_benchmark.Benchmark):
    def params(self):
        self.default('number of neurons', n_neurons=4000)
        self.default('number of dimensions', D=2)
        self.default('maximum firing rate', max_rate=80)
        self.default('stimulus strength', stim_mag=1.4)
        self.default('amount of neuron noise', neuron_noise=0.009)
        self.default('additive bias', neuron_bias=0.0)  # -0.04, 0.046
        self.default('multiplicative gain', neuron_gain=1.0)  # 1.036, 0.960
        self.default('recurrent synapse', synapse_memory=0.1)
        self.default('memory feedback decay value', memory_decay=1.0)
        self.default('do curve fitting', do_fit=False)
        self.default('number of trials to average over', n_trials=1)
        self.default('empirical dataset', dataset='pre_PHE')
        self.default('plot type', plot_type='all')
        self.default('noise of memory estimation', noise_readout=0.23)
        self.default('misperception prob', misperceive=0.066)
        self.default('simulation time', simtime=10.0)
        self.default('ramp input scale', ramp_scale=0.18)
        self.default('analyze spike data', analyze_spikes=False)

    def model(self, p, probe_spikes=False):
        model = nengo.Network()
        model.config[nengo.Ensemble].max_rates = \
            nengo.dists.Uniform(p.max_rate / 2, p.max_rate)

        with model:
            stim = nengo.Node(lambda t: 1 if 0 < t < 1 else 0)
            ramp = nengo.Node(lambda t: t > 1)

            sensory = EnsembleArray(n_neurons=100, n_ensembles=p.D)

            model.cur_node = CurrentNode(np.zeros((p.n_neurons, p.D)))

            memory = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D)
            memory.noise = \
                nengo.processes.WhiteNoise(
                    dist=nengo.dists.Gaussian(mean=0.0,
                                              std=p.neuron_noise))
            model.memory = memory

            nengo.Connection(stim, sensory.input[0], synapse=None)
            nengo.Connection(ramp, sensory.input[1], synapse=None)
            nengo.Connection(sensory.output[0], model.cur_node[0],
                             synapse=p.synapse_memory,
                             transform=p.stim_mag * p.synapse_memory)
            nengo.Connection(sensory.output[1], model.cur_node[1],
                             synapse=p.synapse_memory,
                             transform=p.synapse_memory * p.ramp_scale)

            nengo.Connection(memory, model.cur_node, synapse=p.synapse_memory,
                             transform=p.memory_decay)

            nengo.Connection(model.cur_node, memory.neurons, synapse=None)

            self.p_mem = nengo.Probe(memory, synapse=0.1, sample_every=0.5)
            self.p_mem2 = nengo.Probe(memory, synapse=0.1, sample_every=0.1)

            if probe_spikes:
                self.p_spikes = nengo.Probe(memory.neurons)
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
            return (scipy.stats.norm.cdf(x / noise) * (1 - ignore)
                    + 0.5 * ignore)

        for trial in range(p.n_trials):
            model = self.model(p, probe_spikes=(trial == (p.n_trials - 1) and
                                                p.analyze_spikes))
            model.seed = p.seed + trial

            sim = nengo.Simulator(model)

            model.cur_node.set_encoders(sim.data[model.memory].encoders)
            model.cur_node.bias = p.neuron_bias
            model.cur_node.gain = p.neuron_gain

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

        if p.analyze_spikes:
            spike_data = sim.data[self.p_spikes]
            smoothed_spike_data = np.zeros(spike_data.shape)
            print smoothed_spike_data.shape

            sigma = 0.02
            t_width = 0.2
            t_h = np.arange(t_width / p.dt) * p.dt - t_width / 2.0
            h = np.exp(-t_h ** 2 / (2 * sigma ** 2))
            h = h / np.linalg.norm(h, 1)

            t_interest = [sim.trange() > 1.5]
            nn_interest = []

            for nn in range(spike_data.shape[1]):
                smoothed_data = np.convolve(spike_data[:, nn], h, mode='same')
                smoothed_spike_data[:, nn] = smoothed_data

                smoothed_interest = smoothed_spike_data[:, nn][t_interest]
                deriv = np.mean(np.diff(smoothed_interest) / p.dt)

                # preferred direction maybe?
                if np.mean(smoothed_interest) > 30 and deriv > 0 and \
                   deriv < 0.5:
                    nn_interest.append(nn)

            print "DONE PROCESSING SPIKES, %d" % len(nn_interest)

        if plt is not None:
            num_plots = 3 + p.analyze_spikes

            plt.subplot(num_plots, 1, 1)
            plt.fill_between([2, 4, 6, 8], ci_0_model_results,
                             ci_1_model_results, color='#aaaaaa')
            plt.plot([2, 4, 6, 8], curve_results,
                     label='model ($\sigma$=%0.2f, ig=%1.3f)' % (cp[0], cp[1]))
            plt.plot([2, 4, 6, 8], exp_data, label='exp data')
            plt.legend(loc='best')

            plt.subplot(num_plots, 1, 2)
            plt.fill_between(sim.trange(self.p_mem2.sample_every),
                             ci_integrator_values[:, 0],
                             ci_integrator_values[:, 1], color='#aaaaaa')
            plt.plot(sim.trange(self.p_mem2.sample_every),
                     mean_integrator_values)

            plt.subplot(num_plots, 1, 3)
            plt.plot(sim.trange(self.p_mem2.sample_every),
                     sim.data[self.p_mem2])

            if p.analyze_spikes and len(nn_interest) > 0:
                print ">>", nn_interest[0], \
                    np.mean(smoothed_spike_data[:, nn_interest[0]][t_interest])
                plt.subplot(num_plots, 1, 4)
                plt.plot(sim.trange(), smoothed_spike_data[:, nn_interest[0]])
            if p.analyze_spikes and len(nn_interest) <= 0:
                print "NO SPIKES TO PLOT"

        print "DONE PLOTTING"

        return dict(rmse=np.sqrt(np.mean(curve_results - exp_data) ** 2),
                    choice_noise=cp[0],
                    ignore=cp[1],
                    values=values.tolist())

if __name__ == '__main__':
    BioSpaunMemory().run()
