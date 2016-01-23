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
        self.default('stimulus strength', stim_mag=1.0)
        self.default('amount of neuron noise', neuron_noise=0.01)
        self.default('recurrent synapse', synapse_memory=0.1)
        self.default('averages', averages=2)
        self.default('mean memory noise', mean_mem_noise=0.0)
        self.default('empirical dataset', dataset='pre_PHE')
        self.default('noise of memory estimation', noise_readout=0.1) #near the optimized value
        self.default('plot type', plot_type='all')

    def model(self, p):
        model = nengo.Network()
        model.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(p.max_rate/2, 
                                                                     p.max_rate)
        with model:
            stim = nengo.Node(lambda t: p.stim_mag if 0<t<1 else 0)

            sensory = nengo.Ensemble(n_neurons=100, dimensions=1)

            memory = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D)
            memory.noise = nengo.processes.WhiteNoise(
                                            dist=nengo.dists.Gaussian(mean=p.mean_mem_noise, 
                                                   std=p.neuron_noise))

            nengo.Connection(stim, sensory, synapse=None)
            nengo.Connection(sensory, memory[0], synapse=0.01)
            nengo.Connection(memory, memory, synapse=p.synapse_memory)

            self.p_mem = nengo.Probe(memory, synapse=0.1, sample_every=0.5)
            self.p_spikes = nengo.Probe(memory.neurons)
            self.memory = memory

        return model

    def evaluate_model(self, p, Simulator, model, plt):
        model_data=[]
        spikes_preferred=[]
        spikes_nonpreferred=[]
        rng = np.random.RandomState(seed=p.seed)
        for a in range(p.averages):
            model.seed=rng.randint(0x7FFFFFFF)
            sim = Simulator(model, seed=rng.randint(0x7FFFFFFF))
            sim.run(9)
            model_data.append(sim.data[self.p_mem][[5,9,13,17],0])
            enc=sim.data[self.memory].encoders
            pos=np.where(enc[:,0]>0.2)[0]  #find indices of encoders with +/- preffered direction
            neg=np.where(enc[:,0]<-0.2)[0]  
            spk_pos=sim.data[self.p_spikes][:,pos]  #collect spike data from these indices
            spk_neg=sim.data[self.p_spikes][:,neg]
            spikes_preferred.append(spk_pos)
            spikes_nonpreferred.append(spk_neg)

        self.record_speed(9*p.averages)
        mean_values = np.mean(model_data, axis=0)   # model data
        avg_pref=np.mean(np.hstack(spikes_preferred), axis=1)
        avg_nonpref=np.mean(np.hstack(spikes_nonpreferred), axis=1)
        exp_data_dict = {
            'pre_PHE' : [0.972, 0.947, 0.913, 0.798],  # experimental data from WebPlotDigitizer
            'post_PHE' : [0.972, 0.938, 0.847, 0.666],  # 800-1200 trials
            'pre_GFC' : [0.970, 0.942, 0.882, 0.766],
            'post_GFC' : [0.966, 0.928, 0.906, 0.838],
            }
        exp_data=exp_data_dict[str(p.dataset)]
        ci = np.array([ctn_benchmark.stats.bootstrapci(d, np.mean) for d in np.array(model_data).T])
        dt = 0.001
        sigma = 0.01   #width of smoothing gaussian
        t_h = np.arange(200)*dt-0.1
        h = np.exp(-t_h**2/(2*sigma**2))
        h = h/np.linalg.norm(h,1)
        smoothed_pos=np.convolve(avg_pref,h,mode='same') #convolve with gaussian
        smoothed_neg=np.convolve(avg_nonpref,h,mode='same') 

        def curve(x, noise):
            return scipy.stats.norm.cdf(x/noise)

        rmse = np.sqrt(np.mean((curve(mean_values, p.noise_readout)-exp_data)**2))
        # p, err = scipy.optimize.curve_fit(curve, mean_values, exp_data)
        # rmse = np.sqrt(np.mean((curve(mean_values, *p)-exp_data)**2))

        if plt is not None:
            # plt.fill_between([2,4,6,8], curve(ci[:,0], *p), curve(ci[:,1], *p), color='#aaaaaa')
            # plt.plot([2,4,6,8], curve(mean_values, *p), label='model_data ($\sigma$=%0.2f)' % p[0])

            plt.figure(1)
            plt.fill_between([2,4,6,8], ci[:,0], ci[:,1], color='#aaaaaa')
            plt.plot([2,4,6,8], mean_values, label='model_data')
            plt.xlabel("time (s)")
            plt.ylabel("integrator value")
            plt.ylim(0,p.stim_mag)
            plt.legend(loc='best')
            values=mean_values

            plt.figure(2)
            plt.plot(sim.trange(),smoothed_pos, label='preferred direction')
            plt.plot(sim.trange(),smoothed_neg, label='nonpreferred direction')
            plt.xlabel('time (s)')
            plt.ylabel('normalized firing rate')
            plt.legend(loc='best')
            values=np.array(0)

            plt.figure(4)
            plt.fill_between([2,4,6,8], curve(ci[:,0], p.noise_readout), curve(ci[:,1], p.noise_readout), color='#aaaaaa')
            plt.plot([2,4,6,8], curve(mean_values, p.noise_readout), label='model_data (RMSE=%0.2f)' % rmse)
            plt.plot([2,4,6,8], exp_data, label='exp_data')
            plt.xlabel('time (s)')
            plt.ylabel('percent correct')
            plt.ylim(0.6)
            plt.legend(loc='best')
            values=curve(mean_values, p.noise_readout)

        return dict(rmse=rmse,values=values.tolist(),)

if __name__ == '__main__':
    BioSpaunMemory().run()