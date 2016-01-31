import ctn_benchmark
import nengo
import scipy.optimize
import scipy.stats
import numpy as np
<<<<<<< HEAD
import time

class BioSpaunMemory(ctn_benchmark.Benchmark):
    def params(self):
        self.default('number of neurons', n_neurons=500)
        self.default('number of dimensions', D=32)
        self.default('maximum firing rate', max_rate=80)
        self.default('stimulus strength', stim_mag=1.0)
        self.default('amount of neuron noise', neuron_noise=0.01)
        self.default('recurrent synapse', synapse_memory=0.1)
        self.default('averages', averages=2)
        self.default('mean memory noise', mean_mem_noise=0.0)
        self.default('empirical dataset', dataset='pre_PHE')
        self.default('noise of memory estimation', noise_readout=0.1) #near the optimized value
        self.default('plot type', plot_type='all')
        self.default('misperception prob', misperceive=0.0)
=======

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
>>>>>>> 601d34edf4324e6765afa65b2f06c7596e7f9cc8

    def model(self, p):
        model = nengo.Network()
        model.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(p.max_rate/2, 
                                                                     p.max_rate)
        with model:
<<<<<<< HEAD

            p.stim = nengo.Node(lambda t: p.stim_mag if 0<t<1 else 0)
=======
            stim = nengo.Node(lambda t: 1 if 0<t<1 else 0)
>>>>>>> 601d34edf4324e6765afa65b2f06c7596e7f9cc8

            sensory = nengo.Ensemble(n_neurons=100, dimensions=1)

            memory = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D)
<<<<<<< HEAD
            
            bias_node = nengo.Node(1)
            nengo.Connection(bias_node, memory.neurons, 
                             transform=[[p.mean_mem_noise]] * p.n_neurons)

            nengo.Connection(p.stim, sensory, synapse=None)
            # nengo.Connection(sensory, memory[0], synapse=0.01)
            nengo.Connection(sensory, memory[0], synapse=p.synapse_memory)
            nengo.Connection(memory, memory, synapse=p.synapse_memory)

            # self.p_mem = nengo.Probe(memory, synapse=0.1, sample_every=0.5)
            self.p_mem = nengo.Probe(memory, synapse=0.05, sample_every=0.5)
            self.p_spikes = nengo.Probe(memory.neurons)
            self.memory = memory

        return model

    def evaluate_model(self, p, Simulator, model, plt):
        model_data=[]
        spikes_preferred=[]
        spikes_nonpreferred=[]
        num_wrong = 0

        if p.seed == 0:
            p.seed = int(time.time())

        rng = np.random.RandomState(seed=p.seed)
        for a in range(p.averages):
            model.seed=rng.randint(0x7FFFFFFF)
            #the 'give a shit' or misperception probability
            wrong=np.random.rand()
            if wrong < p.misperceive:
                # p.stim.output = lambda t: -p.stim_mag if 0<t<1 else 0
                num_wrong += 1
                p.stim.output = lambda t: 0 if 0<t<1 else 0
            else:
                p.stim.output = lambda t: p.stim_mag if 0<t<1 else 0
            sim = Simulator(model, seed=rng.randint(0x7FFFFFFF))
            sim.run(9)
            model_data.append(sim.data[self.p_mem][[5,9,13,17],0])
            enc=sim.data[self.memory].encoders
            pos=np.where(enc[:,0]>0.5)[0]  #find indices of encoders with +/- preffered direction
            neg=np.where(enc[:,0]<-0.5)[0]  
            spk_pos=sim.data[self.p_spikes][:,pos]  #collect spike data from these indices
            spk_neg=sim.data[self.p_spikes][:,neg]
            spikes_preferred.append(spk_pos)
            spikes_nonpreferred.append(spk_neg)

        self.record_speed(9*p.averages)
        mean_values = np.mean(model_data, axis=0)   # model data
        ci_values = np.array([ctn_benchmark.stats.bootstrapci(d, np.mean) 
                              for d in np.array(model_data).T])

        def curve(x, noise):
            return scipy.stats.norm.cdf((x)/noise)

        ####################
        print "WRONG: %i of %i, %f" % (num_wrong, p.averages, 1.0 * num_wrong / p.averages)

        model_results = []
        for data in model_data:
            model_results.append(curve(data, p.noise_readout))

        mean_results = np.mean(model_results, axis=0)

        ci_results = np.array([ctn_benchmark.stats.bootstrapci(d, np.mean) 
                               for d in np.array(model_results).T])
        ####################

        avg_pref=np.mean(np.hstack(spikes_preferred), axis=1)
        avg_nonpref=np.mean(np.hstack(spikes_nonpreferred), axis=1)
        exp_data_dict = {
            'pre_PHE' : [0.972, 0.947, 0.913, 0.798],  # experimental data from WebPlotDigitizer
            'post_PHE' : [0.972, 0.938, 0.847, 0.666],  # 800-1200 trials
            'pre_GFC' : [0.970, 0.942, 0.882, 0.766],
            'post_GFC' : [0.966, 0.928, 0.906, 0.838],
            }
        exp_data=exp_data_dict[str(p.dataset)]
        dt = 0.001
        sigma = 0.01   #width of smoothing gaussian
        t_h = np.arange(200)*dt-0.1
        h = np.exp(-t_h**2/(2*sigma**2))
        h = h/np.linalg.norm(h,1)
        smoothed_pos=np.convolve(avg_pref,h,mode='same') #convolve with gaussian
        smoothed_neg=np.convolve(avg_nonpref,h,mode='same') 

        rmse = np.sqrt(np.mean(mean_results-exp_data)**2)
        values=np.array([mean_values, mean_results])
        # p, err = scipy.optimize.curve_fit(curve, mean_values, exp_data)
        # rmse = np.sqrt(np.mean((curve(mean_values, *p)-exp_data)**2))

        if plt is not None:
            # plt.fill_between([2,4,6,8], curve(ci[:,0], *p), curve(ci[:,1], *p), color='#aaaaaa')
            # plt.plot([2,4,6,8], curve(mean_values, *p), label='model_data ($\sigma$=%0.2f)' % p[0])

            plt.figure(1)
            plt.fill_between([2,4,6,8], ci_values[:,0], ci_values[:,1], color='#aaaaaa')
            plt.plot([2,4,6,8], mean_values, label='model_data')
            plt.xlabel("time (s)")
            plt.ylabel("integrator value")
            plt.ylim(0,p.stim_mag)
            plt.legend(loc='best')

            plt.figure(2)
            plt.plot(sim.trange(),smoothed_pos, label='preferred direction')
            plt.plot(sim.trange(),smoothed_neg, label='nonpreferred direction')
            plt.xlabel('time (s)')
            plt.ylabel('normalized firing rate')
            plt.legend(loc='best')

            plt.figure(4)
            plt.fill_between([2,4,6,8], ci_results[:,0], ci_results[:,1], color='#aaaaaa')
            plt.plot([2,4,6,8], mean_results, label='model_data (RMSE=%0.2f)' % rmse)
            plt.plot([2,4,6,8], exp_data, label='exp_data')
            plt.xlabel('time (s)')
            plt.ylabel('percent correct')
            plt.ylim(0.6)
            plt.legend(loc='best')

        return dict(rmse=rmse,values=values.tolist(),)

if __name__ == '__main__':
    BioSpaunMemory().run()
=======
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
>>>>>>> 601d34edf4324e6765afa65b2f06c7596e7f9cc8
