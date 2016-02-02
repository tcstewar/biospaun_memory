import time
import numpy as np
import scipy.optimize
import scipy.stats

import nengo
from nengo.networks import EnsembleArray
import ctn_benchmark


class BioSpaunMemory(ctn_benchmark.Benchmark):
    def params(self):
        self.default('number of neurons', n_neurons=4000)
        self.default('number of dimensions', D=2)
        self.default('maximum firing rate', max_rate=80)
        self.default('stimulus strength', stim_mag=1.2)
        self.default('ramp input scale', ramp_scale=0.2)
        self.default('recurrent synapse', synapse_memory=0.1)

        self.default('memory feedback decay value', memory_decay=1.0)

        self.default('amount of neuron noise', neuron_noise=0.009)

        self.default('additive bias (pre PHE)', neuron_bias_pre_PHE=0.00)
        self.default('additive bias (pre GFC)', neuron_bias_pre_GFC=0.00)
        self.default('additive bias (post PHE)', neuron_bias_post_PHE=-0.0005)
        self.default('additive bias (post GFC)', neuron_bias_post_GFC=0.0002)

        self.default('drug name', drug_name='GFC')

        self.default('min preferred encoder value', min_pref_enc=0.0)
        self.default('max preferred encoder value', max_pref_enc=1.0)
        self.default('min nonpreferred encoder value', min_nonpref_enc=-1.0)
        self.default('max nonpreferred encoder value', max_nonpref_enc=0.0)

        self.default('FIND: Smoothed spike pattern with this minimum slope',
                     find_min_deriv=0.0)
        self.default('FIND: Smoothed spike pattern with this maximum slope',
                     find_max_deriv=0.5)
        self.default('FIND: Smoothed spike pattern with this minimum firing' +
                     ' rate', find_min_rate=30.0)
        self.default('FIND: Smoothed spike pattern with this maximum firing' +
                     ' rate', find_max_rate=80.0)
        self.default('FIND: Smoothed spike pattern with this minimum ' +
                     ' difference between the pre and post drug firing rates',
                     find_min_rate_diff=5.0)
        self.default('FIND: Smoothed spike pattern with this maximum ' +
                     ' difference between the pre and post drug firing rates',
                     find_max_rate_diff=10.0)

        self.default('do curve fitting', do_fit=False)
        self.default('noise of memory estimation', noise_readout=0.19)
        self.default('misperception prob', misperceive=0.066)

        self.default('simulation time', simtime=10.0)

    def model(self, p, neuron_bias=0):
        model = nengo.Network()
        model.config[nengo.Ensemble].max_rates = \
            nengo.dists.Uniform(p.max_rate / 2, p.max_rate)

        with model:
            stim = nengo.Node(lambda t: 1 if 0 < t < 0.5 else 0)
            ramp = nengo.Node(lambda t: t > 0.5)

            sensory = EnsembleArray(n_neurons=100, n_ensembles=p.D)

            memory = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D)
            memory.noise = \
                nengo.processes.WhiteNoise(
                    dist=nengo.dists.Gaussian(mean=neuron_bias,
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

            self.mem_ens = memory
            self.p_mem = nengo.Probe(memory, synapse=0.1, sample_every=0.5)
            self.p_mem2 = nengo.Probe(memory, synapse=0.1, sample_every=0.1)

            self.p_spikes = nengo.Probe(memory.neurons)
        return model

    def get_exp_data(self, dataset):
        exp_data_dict = {
            'pre_PHE': [0.972, 0.947, 0.913, 0.798],  # experimental data from WebPlotDigitizer  # noqa
            'post_PHE': [0.972, 0.938, 0.847, 0.666],  # 800-1200 trials
            'pre_GFC': [0.970, 0.942, 0.882, 0.766],
            'post_GFC': [0.966, 0.928, 0.906, 0.838]}

        return exp_data_dict[dataset]

    def get_dataset_name(self, prefix, drug_name):
        return prefix + '_' + drug_name

    def evaluate(self, p, sim, plt):
        if p.seed == 0:
            p.seed = int(time.time())

        def curve(x, noise, ignore):
            return (scipy.stats.norm.cdf(x / noise) * (1 - ignore)
                    + 0.5 * ignore)

        model_results = {}
        exp_data = {}
        integrator_values = {}
        spike_datas = {}
        encoder_data = {}

        # Spiking data smoothing filter
        sigma = 0.04
        t_width = 0.2
        t_h = np.arange(t_width / p.dt) * p.dt - t_width / 2.0
        h = np.exp(-t_h ** 2 / (2 * sigma ** 2))
        h = h / np.linalg.norm(h, 1)

        for prefix in ['pre', 'post']:
            dataset = self.get_dataset_name(prefix, p.drug_name)
            neuron_bias = getattr(p, 'neuron_bias_' + dataset)

            model = self.model(p, neuron_bias)
            model.seed = p.seed

            sim = nengo.Simulator(model, seed=p.seed)
            sim.run(p.simtime)
            self.record_speed(p.simtime)

            values = sim.data[self.p_mem][[5, 9, 13, 17], 0]   # model data
            integrator_value = sim.data[self.p_mem2]

            # Process model data
            model_results[dataset] = values
            integrator_values[dataset] = integrator_value
            exp_data[dataset] = self.get_exp_data(dataset)
            encoder_data[dataset] = sim.model.params[self.mem_ens].encoders

            # Process spike data
            spike_data = sim.data[self.p_spikes]
            smoothed_spike_data = np.zeros(spike_data.shape)

            for nn in range(spike_data.shape[1]):
                smoothed_data = np.convolve(spike_data[:, nn], h, mode='same')
                smoothed_spike_data[:, nn] = smoothed_data
            spike_datas[dataset] = smoothed_spike_data

        if p.do_fit:
            # TODO: FIX CURVE FIT
            cp = [p.noise_readout, p.misperceive]
            # cp, err = scipy.optimize.curve_fit(curve, mean_model_results,
            #                                    exp_data)
        else:
            cp = [p.noise_readout, p.misperceive]

        # Process pre/post drug spike data
        find_min_deriv_pre = p.find_min_deriv
        find_max_deriv_pre = p.find_max_deriv
        min_rate_pre = p.find_min_rate
        max_rate_pre = p.find_max_rate

        t_interest = [(sim.trange() > 2.0) & (sim.trange() < (p.simtime - 2))]
        nn_interest_pre_preferred = []
        nn_interest_pre_nonpreferred = []

        smoothed_data_pre = \
            spike_datas[self.get_dataset_name('pre', p.drug_name)]

        for nn in range(smoothed_data_pre.shape[1]):
            smoothed_interest = smoothed_data_pre[:, nn][t_interest]
            deriv = np.mean(np.diff(smoothed_interest) / p.dt)

            # Optional deriviative calculation
            # deriv = ((smoothed_interest[-1] - smoothed_interest[0]) /
            #          (sim.trange()[t_interest][-1] -
            #           sim.trange()[t_interest][0]))
            enc_dir = encoder_data[self.get_dataset_name('pre', p.drug_name)][nn, :][0]
            if np.mean(smoothed_interest) > min_rate_pre and \
               np.mean(smoothed_interest) < max_rate_pre and \
               deriv > find_min_deriv_pre and deriv < find_max_deriv_pre and \
               enc_dir > p.min_pref_enc and \
               enc_dir < p.max_pref_enc:
                    nn_interest_pre_preferred.append(nn)
            if np.mean(smoothed_interest) > min_rate_pre and \
               np.mean(smoothed_interest) < max_rate_pre and \
               deriv > find_min_deriv_pre and deriv < find_max_deriv_pre and \
               enc_dir > p.min_nonpref_enc and \
               enc_dir < p.max_nonpref_enc:
                    nn_interest_pre_nonpreferred.append(nn)

        min_pre_post_rate_diff = p.find_min_rate_diff
        max_pre_post_rate_diff = p.find_max_rate_diff

        nn_interest_post_preferred = []
        nn_interest_post_nonpreferred = []
        smoothed_data_post = \
            spike_datas[self.get_dataset_name('post', p.drug_name)]

        for nn in nn_interest_pre_preferred:
            smoothed_interest_pre = smoothed_data_pre[:, nn][t_interest]
            ave_rate_pre = np.mean(smoothed_interest_pre)

            smoothed_interest_post = smoothed_data_post[:, nn][t_interest]
            ave_rate_post = np.mean(smoothed_interest_post)

            # print '><', ave_rate_pre, ave_rate_post,\
            #     encoder_data[self.get_dataset_name('pre', p.drug_name)][nn, :]

            if ave_rate_post - ave_rate_pre > min_pre_post_rate_diff and \
               ave_rate_post - ave_rate_pre < max_pre_post_rate_diff:
                nn_interest_post_preferred.append(nn)
            #     print "!"

        for nn in nn_interest_pre_nonpreferred:
            smoothed_interest_pre = smoothed_data_pre[:, nn][t_interest]
            ave_rate_pre = np.mean(smoothed_interest_pre)

            smoothed_interest_post = smoothed_data_post[:, nn][t_interest]
            ave_rate_post = np.mean(smoothed_interest_post)

            # print '><', ave_rate_pre, ave_rate_post,\
            #     encoder_data[self.get_dataset_name('pre', p.drug_name)][nn, :]

            if ave_rate_post - ave_rate_pre > min_pre_post_rate_diff and \
               ave_rate_post - ave_rate_pre < max_pre_post_rate_diff:
                nn_interest_post_nonpreferred.append(nn)
            #     print "!"

        print "number of chosen preferred neurons, pre and post >>", \
          len(nn_interest_pre_preferred), len(nn_interest_post_preferred)
        print "number of chosen nonpreferred neurons, pre and post >>", \
          len(nn_interest_pre_nonpreferred), len(nn_interest_post_nonpreferred)

        if plt is not None:
            for prefix in ['pre', 'post']:
                dataset = self.get_dataset_name(prefix, p.drug_name)
                curve_result = curve(model_results[dataset], *cp)

                plt.subplot(2, 1, 1)
                plt.plot([2, 4, 6, 8], curve_result)
                plt.plot([2, 4, 6, 8], exp_data[dataset], '--')
                plt.legend(['pre_%s model' % p.drug_name,
                            'pre_%s data' % p.drug_name,
                            'post_%s model' % p.drug_name,
                            'post_%s data' % p.drug_name])
                plt.title('Accuracy')

                plt.subplot(2, 1, 2)
                plt.plot(sim.trange(self.p_mem2.sample_every),
                         integrator_values[dataset])
                plt.legend(['pre_%s[0]' % p.drug_name,
                            'pre_%s[1]' % p.drug_name,
                            'post_%s[0]' % p.drug_name,
                            'post_%s[1]' % p.drug_name])
                plt.title('Integrator value')
                
            thefontsize=42
            thelinewidth=7
            if len(nn_interest_post_preferred) > 0:
                # Mean smoothed rates for preferred direction neurons
                plt.figure(figsize=(16, 8))
                plt.plot(sim.trange(),
                         np.mean(smoothed_data_pre[:, nn_interest_post_preferred], axis=1),
                         color='b',linewidth=thelinewidth, label='Control')
                plt.plot(sim.trange(),
                         np.mean(smoothed_data_post[:, nn_interest_post_preferred], axis=1),
                         color='r',linewidth=thelinewidth, label='%s' % p.drug_name)
                plt.fill_between([0.0,0.5],[0,0],[plt.ylim()[1],plt.ylim()[1]], color='darkgray')
                plt.fill_between([0.5,3.0],[0,0],[plt.ylim()[1],plt.ylim()[1]], color='lightgray')
                plt.legend(loc='lower right',fontsize=thefontsize)
                label_xticks = ['Cue','0.5','Delay','3']
                plt.xticks([0.2,0.5,1.75,3],label_xticks,fontsize=thefontsize)
                plt.yticks(fontsize=thefontsize)
                plt.xlim(0,3)
                plt.ylim(0,15)
                # plt.xlabel('time (s)',fontsize=thefontsize)
                plt.ylabel('Normalized Firing Rate',fontsize=thefontsize)
                # plt.title('Preferred Direction',fontsize=thefontsize)

            if len(nn_interest_post_nonpreferred) > 0:
                # Mean smoothed rates for nonpreferred direction neurons
                plt.figure(figsize=(16, 8))
                plt.plot(sim.trange(),
                         np.mean(smoothed_data_pre[:, nn_interest_post_nonpreferred], axis=1),
                         color='b',linewidth=thelinewidth, label='Control')
                plt.plot(sim.trange(),
                         np.mean(smoothed_data_post[:, nn_interest_post_nonpreferred], axis=1),
                         color='r',linewidth=thelinewidth, label='%s' % p.drug_name)
                plt.fill_between([0.0,0.5],[0,0],[plt.ylim()[1],plt.ylim()[1]], color='darkgray')
                plt.fill_between([0.5,3.0],[0,0],[plt.ylim()[1],plt.ylim()[1]], color='lightgray')
                plt.legend(loc='lower right',fontsize=thefontsize)
                label_xticks = ['Cue','0.5','Delay','3']
                plt.xticks([0.2,0.5,1.75,3],label_xticks,fontsize=thefontsize)
                plt.yticks(fontsize=thefontsize)
                plt.xlim(0,3)
                plt.ylim(0,15)
                # plt.xlabel('time (s)',fontsize=thefontsize)
                plt.ylabel('Normalized Firing Rate',fontsize=thefontsize)
                # plt.title('Nonpreferred Direction',fontsize=thefontsize)

        print "S>", p.seed

        return dict(choice_noise=cp[0],
                    ignore=cp[1],
                    seed=p.seed)

if __name__ == '__main__':
    BioSpaunMemory().run()
