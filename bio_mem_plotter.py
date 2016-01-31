#bio_mem4_plotter

import numpy as np
import matplotlib.pyplot as plt


#Load em up
baseline=np.load('pre_GFC3.npz')
baseline_experiment_x=baseline['experiment_x']
baseline_integrator_x=baseline['integrator_x']
baseline_model_x=baseline['model_x']
baseline_experiment_y=baseline['experiment_y']
baseline_integrator_y=baseline['integrator_y']
baseline_model_y=baseline['model_y']
baseline_integrator_ci=baseline['integrator_ci']
baseline_model_ci=baseline['model_ci']

post_gfc=np.load('post_GFC1.npz')
post_gfc_experiment_x=post_gfc['experiment_x']
post_gfc_integrator_x=post_gfc['integrator_x']
post_gfc_model_x=post_gfc['model_x']
post_gfc_experiment_y=post_gfc['experiment_y']
post_gfc_integrator_y=post_gfc['integrator_y']
post_gfc_model_y=post_gfc['model_y']
post_gfc_integrator_ci=post_gfc['integrator_ci']
post_gfc_model_ci=post_gfc['model_ci']

post_phe=np.load('post_PHE3.npz')
post_phe_experiment_x=post_phe['experiment_x']
post_phe_integrator_x=post_phe['integrator_x']
post_phe_model_x=post_phe['model_x']
post_phe_experiment_y=post_phe['experiment_y']
post_phe_integrator_y=post_phe['integrator_y']
post_phe_model_y=post_phe['model_y']
post_phe_integrator_ci=post_phe['integrator_ci']
post_phe_model_ci=post_phe['model_ci']



#Plot Choice Figure
thefontsize=32
thelinewidth=5
plt.figure(figsize=(18, 10))

plt.plot(baseline_experiment_x, baseline_experiment_y, linewidth=thelinewidth,
	label='Experiment Baseline',color='b')
plt.plot(baseline_model_x, baseline_model_y, linewidth=thelinewidth, linestyle='dashed', \
        label='Model Baseline (RMSE = %f)'
        % np.sqrt(np.mean(baseline_model_y - baseline_experiment_y) ** 2),color='b')
plt.fill_between(baseline_model_x, baseline_model_ci[0], baseline_model_ci[1], color='#aaaaaa')

plt.plot(post_gfc_experiment_x, post_gfc_experiment_y, linewidth=thelinewidth,
	label='Experiment GFC',color='r')
plt.plot(post_gfc_model_x, post_gfc_model_y, linewidth=thelinewidth, linestyle='dashed', \
        label='Model GFC (RMSE = %f)'
        % np.sqrt(np.mean(post_gfc_model_y - post_gfc_experiment_y) ** 2),color='r')
plt.fill_between(post_gfc_model_x, post_gfc_model_ci[0], post_gfc_model_ci[1], color='#aaaaaa')

plt.plot(post_phe_experiment_x, post_phe_experiment_y, linewidth=thelinewidth,
	label='Experiment PHE',color='g')
plt.plot(post_phe_model_x, post_phe_model_y, linewidth=thelinewidth, linestyle='dashed', \
        label='Model PHE (RMSE = %f)'
        % np.sqrt(np.mean(post_phe_model_y - post_phe_experiment_y) ** 2),color='g')
plt.fill_between(post_phe_model_x, post_phe_model_ci[0], post_phe_model_ci[1], color='#aaaaaa')

plt.xticks(fontsize=thefontsize)
plt.yticks(fontsize=thefontsize)
plt.xlabel('Delay Period Length (s)',fontsize=thefontsize)
plt.ylabel('DRT Accuracy (% correct)',fontsize=thefontsize)
plt.legend(loc='lower left',fontsize=thefontsize-4)


#Plot Integrator Figure
plt.figure(figsize=(16, 8))

plt.plot(baseline_integrator_x,baseline_integrator_y,linewidth=thelinewidth, label='Baseline', color='b')
plt.fill_between(baseline_integrator_x,baseline_integrator_ci[0],baseline_integrator_ci[1], color='#aaaaaa')

plt.plot(post_gfc_integrator_x,post_gfc_integrator_y,linewidth=thelinewidth, label='GFC', color='r')
plt.fill_between(post_gfc_integrator_x,post_gfc_integrator_ci[0],post_gfc_integrator_ci[1], color='#aaaaaa')

plt.plot(post_phe_integrator_x,post_phe_integrator_y,linewidth=thelinewidth, label='PHE', color='g')
plt.fill_between(post_phe_integrator_x,post_phe_integrator_ci[0],post_phe_integrator_ci[1], color='#aaaaaa')

plt.ylim(0,1.2)
plt.fill_between([0.0,1.0],np.array([0.0,0.0]), \
    np.array([plt.ylim()[1],plt.ylim()[1]]), color='#aaaaaa') #throws a known error
label_xticks = ['Cue','Delay','2','4','6','8']
plt.xticks([0.5,2,3,5,7,9],label_xticks,fontsize=thefontsize)
plt.yticks(fontsize=thefontsize)
plt.xlim(0,9)
# plt.xticks(fontsize=thefontsize)
# plt.yticks(fontsize=thefontsize)
# plt.xlabel('Delay Period Length (s)',fontsize=thefontsize)
plt.ylabel('Integrator Value',fontsize=thefontsize)
plt.legend(loc='uppper right',fontsize=thefontsize)

plt.show()