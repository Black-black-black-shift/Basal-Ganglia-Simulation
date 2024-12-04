import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import time

device = 'cuda'

dt = torch.tensor(0.1, dtype=torch.float32, device=device)
duration = torch.tensor(1000., dtype=torch.float32, device=device)
timesteps = int((duration / dt).item())
time_array = torch.arange(timesteps, dtype=torch.float32, device=device) * dt

cp1 = torch.tensor(0.05, dtype=torch.float32, device=device)
cp2 = torch.tensor(0.02, dtype=torch.float32, device=device)
cp3 = torch.tensor(1, dtype=torch.float32, device=device)
cp4 = torch.tensor(0.05, dtype=torch.float32, device=device)

num_scale = 2500
num_C = 250
num_b = 10

samples = [0, 1, 2]

g_X_C_AMPA = 4.
g_C_C_GABA = 1.0

g_S_G_AMPA = 0.05
g_S_G_NMDA = 2.2
g_b_G_AMPA = 0.1
g_b_G_GABA = 8.
g_C_G_GABA = 0.
g_G_G_GABA = 5.5

g_G_S_GABA = 5.0
g_b_S_AMPA = 0.75

i_rate_X_C_AMPA = 1000
m_rate_X_C_AMPA = 2000
rate_b_G_AMPA = 4000
rate_b_G_GABA = 2000
rate_b_S_AMPA = 3200

scale_d = 1
scale_r = 0.1
delay = int(1/dt)*dt
p_D = 0.
T_ref = 2
window = 5
t_sat = len(time_array)

# Record the start time
#start_time = time.time()

sys.path.append('/public/home/ssct004t/project/Junting')

# Create subdirectory if it doesn't exist
output_dir = 'plots_2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Help Functions
# Poisson spike train generator
def generate_poisson_spike_train(rate, timesteps, dt, device):
    return torch.rand(timesteps, device=device) < rate * dt * 0.001  # rate in Hz, timesteps in ms

def get_psd(v_values, dt):

    # Convert voltage values to a tensor for FFT analysis
    v_tensor = v_values[int(400/dt):].clone()
    v_tensor = v_tensor - torch.mean(v_tensor)

    N = len(v_tensor)
    # Perform FFT and calculate the power spectrum
    v_fft = torch.fft.fft(v_tensor)
    power_spectrum = torch.abs(v_fft) ** 2
    v_psd = power_spectrum / (N * (dt/1000))
    v_freqs = torch.fft.fftfreq(len(v_tensor), dt / 1000)  # Frequency axis in Hz

    return v_psd, v_freqs

def create_ramping(x_s, y_i, y_m, x_array):
    # Ensure x_s and y_s are tensors
    x_s = torch.as_tensor(x_s, dtype=x_array.dtype, device=x_array.device)
    y_i = torch.as_tensor(y_i, dtype=x_array.dtype, device=x_array.device)
    y_m = torch.as_tensor(y_m, dtype=x_array.dtype, device=x_array.device)

    y_array = torch.zeros_like(x_array, device=x_array.device)

    for i, x in enumerate(x_array):
        y_array[i] = y_i + x_array[i] * (y_m - y_i) / x_s if x < x_s else 0
    
    return y_array

def window_rate(values, window, dt):
    win_step = int(window / dt)
    m, t = values.shape
    n = t - win_step
    
    # Create a tensor for the sliding window sums
    cumsum_values = values.cumsum(dim=1)
    
    # Calculate the sum for each window using cumulative sums
    win_sums = cumsum_values[:, win_step:] - cumsum_values[:, :-win_step]
    
    # Compute the rate
    rate = win_sums.sum(dim=0) / (m * win_step * dt)
    
    return rate

#Neuron and Synapse Models

class NeuronGroup(nn.Module):
    def __init__(self, dt, device='cpu', C = 0.5, v_L=-70, v_h=-60, v_T=120, g_L=0.025, g_T=0.06, v_b=-50, v_r=-55, tau_hm=20, tau_hp=100, num=100, T_ref=2):
        super(NeuronGroup, self).__init__()
        # Initialize neuron populations
        self.device = device
        self.C = torch.tensor(C, dtype=torch.float32, device=self.device).clone().detach()
        self.v_r = torch.tensor(v_r, dtype=torch.float32, device=self.device).clone().detach()
        self.v_b = torch.tensor(v_b, dtype=torch.float32, device=self.device).clone().detach()
        self.v_L = torch.tensor(v_L, dtype=torch.float32, device=self.device).clone().detach()
        self.v_h = torch.tensor(v_h, dtype=torch.float32, device=self.device).clone().detach()
        self.v_T = torch.tensor(v_T, dtype=torch.float32, device=self.device).clone().detach()
        self.g_L = torch.tensor(g_L, dtype=torch.float32, device=self.device).clone().detach()
        self.g_T = torch.tensor(g_T, dtype=torch.float32, device=self.device).clone().detach()
        self.tau_hm = torch.tensor(tau_hm, dtype=torch.float32, device=self.device).clone().detach()
        self.tau_hp = torch.tensor(tau_hp, dtype=torch.float32, device=self.device).clone().detach()
        self.T_ref = torch.tensor(T_ref, dtype=torch.float32, device=self.device).clone().detach()
        self.dt = dt
        self.num = num
        self.v = self.v_L.clone().detach().float() * torch.ones(num, device=self.device)
        self.h = torch.ones(num, dtype=torch.float32, device=self.device)
        self.last_spike = torch.zeros(num, dtype=torch.float32, device=self.device)
        self.spike_trains = []

    def syn_connect(self, syns_AMPA=None, syns_NMDA=None, syns_GABA=None):
    #     # avoid repeatedly copy big matrix in syn class
        
        # Initialize synaptic inputs to empty lists if None
        if syns_AMPA is None:
            syns_AMPA = []
        if syns_NMDA is None:
            syns_NMDA = []
        if syns_GABA is None:
            syns_GABA = []
    
        self.syns_AMPA = syns_AMPA
        self.syns_NMDA = syns_NMDA
        self.syns_GABA = syns_GABA

    def forward(self, Time, I_ext=None): 

        not_saturated = Time > self.last_spike + self.T_ref

        if I_ext==None:
            I_ext = torch.zeros(self.num, dtype=torch.float32, device=self.device)
        assert I_ext.shape[0] == self.num

        I_syn = torch.zeros(self.num, dtype=torch.float32, device=self.device)

        for syn in self.syns_AMPA:
            I_syn += syn.g * (self.v - syn.v_rev) * syn.f_all 
        
        for syn in self.syns_NMDA:
            I_syn += syn.g * (self.v - syn.v_rev) * syn.f_all / (1 + torch.exp(-0.062 * self.v / 3.57))

        for syn in self.syns_GABA:
            I_syn += syn.g * (self.v - syn.v_rev) * syn.f_all

        I_syn = torch.where(not_saturated, I_syn, torch.zeros(self.num, dtype=torch.float32, device=self.device))

        T_open = self.v >= self.v_h
        dh1 = -self.h / self.tau_hm * self.dt
        dh2 = (1 - self.h) / self.tau_hp * self.dt
        dh = torch.where(T_open, dh1, dh2)

        dv1 = (-self.g_L * (self.v - self.v_L) - self.g_T * self.h * self.H(self.v - self.v_h) * (self.v -self.v_T) - I_syn + I_ext) / self.C * self.dt
        dv2 = -self.g_L * (self.v - self.v_L) / self.C * self.dt
        dv = torch.where(not_saturated, dv1, dv2)
        
        self.h += dh
        self.v += dv
        
        spike = (self.v >= self.v_b).float()
        v_this = torch.where(spike > 0, self.v_b, self.v)
        self.v = torch.where(spike > 0, self.v_r, self.v)

        self.last_spike = torch.where(spike > 0, Time, self.last_spike)
        #print(-self.g_L * (self.v - self.v_L) / self.C * self.dt, I_ext / self.C * self.dt, self.v)

        self.spike_trains.append(spike)

        return spike, v_this, I_syn
    
    def H(self, x):
        """Custom Heaviside step function integrated within H."""
        # Ensure value_at_zero is on the same device as x
        return torch.where(x >= 0, torch.tensor(1.0, dtype=torch.float32, device=x.device), torch.tensor(0., dtype=torch.float32, device=x.device))
    
class SynapseNetwork(nn.Module):
    def __init__(self, device='cpu', num_a=2500, num_z=2500, type='AMPA', alpha=0.63, conP=0.05, p_D=0, tau_D=600, g=0.05, dt=0.1):
        super(SynapseNetwork, self).__init__()
        self.device = device
        #self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.type = type
        self.g = torch.tensor(g, dtype=torch.float32, device=self.device).clone().detach()
        self.num_a = num_a
        self.num_z = num_z
        self.p_D = torch.tensor(p_D, dtype=torch.float32, device=self.device).clone().detach()
        self.tau_D = torch.tensor(tau_D, dtype=torch.float32, device=self.device).clone().detach()
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device).clone().detach()
        self.dt = dt
        self.conP = conP

        if self.type == 'AMPA':
            self.v_rev = torch.tensor(0, dtype=torch.float32, device=self.device).clone().detach()
            self.tau = torch.tensor(2, dtype=torch.float32, device=self.device).clone().detach()
        elif self.type == 'NMDA':
            self.v_rev = torch.tensor(0, dtype=torch.float32, device=self.device).clone().detach()
            self.tau = torch.tensor(100, dtype=torch.float32, device=self.device).clone().detach()
        elif self.type == 'GABA':
            self.v_rev = torch.tensor(-70, dtype=torch.float32, device=self.device).clone().detach()
            self.tau = torch.tensor(5, dtype=torch.float32, device=self.device).clone().detach()

        self.s = torch.zeros(self.num_a, self.num_z, dtype=torch.float32, device=self.device)
        self.D = torch.ones(self.num_a, self.num_z, dtype=torch.float32, device=self.device)
        self.syns = torch.rand(self.num_a, self.num_z, dtype=torch.float32, device=self.device) < self.conP

        self.f_all = torch.sum(self.syns * self.D * self.s, dim=0)

    def forward(self, spike): 
        
        assert spike.shape[0] == self.num_a
        expanded_spike = spike.unsqueeze(1).expand(-1, self.num_z)

        masked_s = self.s[self.syns]    # Get values of self.s where self.syns is True
        masked_D = self.D[self.syns]
        masked_spike = expanded_spike[self.syns]  # Get spike values for the same positions

        if self.type == 'NMDA':
            ds = self.alpha * (1 - masked_s) * masked_spike - masked_s / self.tau * self.dt
        else:
            ds = masked_spike - masked_s / self.tau * self.dt

        dD = - self.p_D * masked_spike * masked_D + (1 - masked_D) * self.dt / self.tau_D

        self.s[self.syns] += ds
        self.D[self.syns] += dD

        self.f_all = torch.sum(self.syns * self.D * self.s, dim=0)

        return self.s
    
    def update_g(self, new_g):
        """Update conductance `g` for all synapses in the network."""
        
        # Convert new_g to a tensor on the correct device if it's not already a tensor
        if not isinstance(new_g, torch.Tensor):
            new_g = torch.tensor(new_g, dtype=torch.float32, device=self.device).clone().detach()
        else:
            new_g = new_g.to(self.device).clone().detach()
        
        # Update the global g value
        self.g = new_g

    
# Record the start time
start_time = time.time()

#Start simulation

# Instantiate neuron and synapses with the slider values
CD = NeuronGroup(device=device, g_T=0, num=num_C, T_ref=T_ref, dt=dt)
GPe = NeuronGroup(device=device, g_T=0.06, num=num_scale, T_ref=T_ref, dt=dt)
STN = NeuronGroup(device=device, g_T=0.06, num=num_scale, T_ref=T_ref, dt=dt)

v_b = GPe.v_b

syn_X_C_AMPA = SynapseNetwork(device=device, num_a=num_b, num_z=CD.num, type='AMPA', conP=cp3, p_D=p_D, dt=dt)
syn_C_C_GABA = SynapseNetwork(device=device, num_a=CD.num, num_z=CD.num, type='GABA', conP=cp4, dt=dt)

syn_S_G_AMPA = SynapseNetwork(device=device, num_a=STN.num, num_z=GPe.num, type='AMPA', conP=cp1, dt=dt)
syn_S_G_NMDA = SynapseNetwork(device=device, num_a=STN.num, num_z=GPe.num, type='NMDA', conP=cp1, dt=dt)
syn_b_G_AMPA = SynapseNetwork(device=device, num_a=num_b, num_z=GPe.num, type='AMPA', conP=cp3, dt=dt)
syn_b_G_GABA = SynapseNetwork(device=device, num_a=num_b, num_z=GPe.num, type='GABA', conP=cp3, dt=dt)
syn_C_G_GABA = SynapseNetwork(device=device, num_a=CD.num, num_z=GPe.num, type='GABA', conP=cp3, dt=dt)
syn_G_G_GABA = SynapseNetwork(device=device, num_a=GPe.num, num_z=GPe.num, type='GABA', conP=cp1, dt=dt)

syn_G_S_GABA = SynapseNetwork(device=device, num_a=GPe.num, num_z=STN.num, type='GABA', conP=cp2, p_D=p_D, dt=dt)
syn_b_S_AMPA = SynapseNetwork(device=device, num_a=num_b, num_z=STN.num, type='AMPA', conP=cp3, p_D=p_D, dt=dt)

CD.syn_connect(syns_AMPA=[syn_X_C_AMPA], syns_GABA=[syn_C_C_GABA])
GPe.syn_connect(syns_AMPA=[syn_S_G_AMPA, syn_b_G_AMPA], syns_NMDA=[syn_S_G_NMDA], syns_GABA=[syn_b_G_GABA, syn_C_G_GABA, syn_G_G_GABA])
STN.syn_connect(syns_AMPA=[syn_b_S_AMPA], syns_GABA=[syn_G_S_GABA])

rate_X_C_AMPA = create_ramping(t_sat, i_rate_X_C_AMPA, m_rate_X_C_AMPA, time_array)

def simulate_neuron_network(g_X_C_AMPA, g_C_C_GABA, g_S_G_AMPA, g_S_G_NMDA, g_b_G_AMPA, g_b_G_GABA, g_C_G_GABA, g_G_G_GABA, g_G_S_GABA, g_b_S_AMPA, rate_X_C_AMPA, rate_b_G_AMPA, rate_b_G_GABA, rate_b_S_AMPA, scale_d, scale_r, delay):
    
    syn_X_C_AMPA.update_g(g_X_C_AMPA * scale_d * 0.001)
    syn_C_C_GABA.update_g(g_C_C_GABA * scale_d * 0.001)
    
    syn_S_G_AMPA.update_g(g_S_G_AMPA * scale_d * 0.001)
    syn_S_G_NMDA.update_g(g_S_G_NMDA * scale_d * 0.001)
    syn_b_G_AMPA.update_g(g_b_G_AMPA * scale_d * 0.001)
    syn_b_G_GABA.update_g(g_b_G_GABA * scale_d * 0.001)
    syn_C_G_GABA.update_g(g_C_G_GABA * scale_d * 0.001)
    syn_G_G_GABA.update_g(g_G_G_GABA * scale_d * 0.001)

    syn_G_S_GABA.update_g(g_G_S_GABA * scale_d * 0.001)
    syn_b_S_AMPA.update_g(g_b_S_AMPA * scale_d * 0.001)

    rate_X_C_AMPA = torch.tensor(rate_X_C_AMPA, device=device)
    rate_b_G_AMPA = torch.tensor(rate_b_G_AMPA, device=device)
    rate_b_G_GABA = torch.tensor(rate_b_G_GABA, device=device)
    rate_b_S_AMPA = torch.tensor(rate_b_S_AMPA, device=device)
    scale_d = torch.tensor(scale_d, device=device)
    scale_r = torch.tensor(scale_r, device=device)
    delay = delay.to(device).clone().detach()

    d_step = int((delay / dt).item())

    v_CD_values = []
    spike_CD_output = []
    I_CD_values = []

    v_GPe_values = []
    spike_GPe_output = []
    I_GPe_values = []

    v_STN_values = []
    spike_STN_output = []
    I_STN_values = []
	
    # Run the simulation over time
    for t in range(timesteps):

        s_X_C_AMPA = syn_X_C_AMPA((torch.rand(num_b, device=device) < rate_X_C_AMPA[t] * scale_r * dt * 0.001).float())
        s_C_C_GABA = syn_C_C_GABA(CD.spike_trains[-d_step] if len(CD.spike_trains) > d_step else torch.zeros(CD.num, device=device))

        s_S_G_AMPA = syn_S_G_AMPA(STN.spike_trains[-d_step] if len(STN.spike_trains) > d_step else torch.zeros(STN.num, device=device))
        s_S_G_NMDA = syn_S_G_NMDA(STN.spike_trains[-d_step] if len(STN.spike_trains) > d_step else torch.zeros(STN.num, device=device))
        s_b_G_AMPA = syn_b_G_AMPA((torch.rand(num_b, device=device) < rate_b_G_AMPA * scale_r * dt * 0.001).float())
        s_b_G_GABA = syn_b_G_GABA((torch.rand(num_b, device=device) < rate_b_G_GABA * scale_r * dt * 0.001).float())
        s_C_G_GABA = syn_C_G_GABA(CD.spike_trains[-d_step] if len(CD.spike_trains) > d_step else torch.zeros(CD.num, device=device))
        s_G_G_GABA = syn_G_G_GABA(GPe.spike_trains[-d_step] if len(GPe.spike_trains) > d_step else torch.zeros(GPe.num, device=device))

        s_b_S_AMDA = syn_b_S_AMPA((torch.rand(num_b, device=device) < rate_b_S_AMPA * scale_r * dt * 0.001).float())
        s_G_S_GABA = syn_G_S_GABA(GPe.spike_trains[-d_step] if len(GPe.spike_trains) > d_step else torch.zeros(GPe.num, device=device))

        spike_CD, v_CD, I_CD = CD(Time=t*dt)
        spike_GPe, v_GPe, I_GPe = GPe(Time=t*dt)
        spike_STN, v_STN, I_STN = STN(Time=t*dt)

        v_CD_values.append(v_CD.clone())
        spike_CD_output.append(spike_CD.clone())
        I_CD_values.append(I_CD.clone())

        v_GPe_values.append(v_GPe.clone())
        spike_GPe_output.append(spike_GPe.clone())
        I_GPe_values.append(I_GPe.clone())

        v_STN_values.append(v_STN.clone())
        spike_STN_output.append(spike_STN.clone())
        I_STN_values.append(I_STN.clone())
        
        print(t)

    v_CD_values = torch.stack(v_CD_values).T
    spike_CD_output = torch.stack(spike_CD_output).T
    I_CD_values = torch.stack(I_CD_values).T

    v_GPe_values = torch.stack(v_GPe_values).T
    spike_GPe_output = torch.stack(spike_GPe_output).T
    I_GPe_values = torch.stack(I_GPe_values).T

    v_STN_values = torch.stack(v_STN_values).T
    spike_STN_output = torch.stack(spike_STN_output).T
    I_STN_values = torch.stack(I_STN_values).T

    return v_CD_values, spike_CD_output, I_CD_values, v_GPe_values, spike_GPe_output, I_GPe_values, v_STN_values, spike_STN_output, I_STN_values

#Start ploting

v_CD_values, spike_CD_output, I_CD_values, v_GPe_values, spike_GPe_output, I_GPe_values, v_STN_values, spike_STN_output, I_STN_values = simulate_neuron_network(g_X_C_AMPA, g_C_C_GABA, g_S_G_AMPA, g_S_G_NMDA, g_b_G_AMPA, g_b_G_GABA, g_C_G_GABA, g_G_G_GABA, g_G_S_GABA, g_b_S_AMPA, rate_X_C_AMPA, rate_b_G_AMPA, rate_b_G_GABA, rate_b_S_AMPA, scale_d, scale_r, delay)

I_GPe_psd, I_GPe_freqs = get_psd(torch.mean(I_GPe_values, dim=0), dt=dt)
I_STN_psd, I_STN_freqs = get_psd(torch.mean(I_STN_values, dim=0), dt=dt)

rate_CD = window_rate(spike_CD_output, window, dt)
rate_GPe = window_rate(spike_GPe_output, window, dt)
rate_STN = window_rate(spike_STN_output, window, dt)

CD_total_rate = torch.sum(spike_CD_output)

p_num = 9
p_i = 1

plt.figure(figsize=(20, p_num * 3))

plt.subplot(p_num, 1, p_i)
plt.plot(time_array.cpu().numpy(), rate_X_C_AMPA.cpu().numpy())
plt.ylabel('Cortex Input Rate')
plt.xlabel('Time (ms)')
plt.grid()
p_i += 1

plt.subplot(p_num, 1, p_i)
plt.plot(time_array[:-int(window/dt)].cpu().numpy(), rate_CD.cpu().numpy(), color='red', label='rate_CD')
plt.plot(time_array[:-int(window/dt)].cpu().numpy(), rate_GPe.cpu().numpy(), color='blue', label='rate_GPe')
plt.plot(time_array[:-int(window/dt)].cpu().numpy(), rate_STN.cpu().numpy(), color='green', label='rate_STN')
plt.ylabel('spiking Rate')
plt.xlabel('Time (ms)')
plt.legend()
plt.grid()
p_i += 1

#CD

plt.subplot(p_num, 1, p_i)
for k in samples:
    line, = plt.plot(time_array.cpu().numpy(), v_CD_values[k].cpu().numpy(), linewidth=1)
    line_color = line.get_color()
    spike_times = time_array[spike_CD_output[k] == 1].cpu().numpy()  # Get times where spikes occur
    for spike_time in spike_times:
        plt.vlines(x=spike_time, ymin=v_b.cpu().numpy(), ymax=v_b.cpu().numpy() + 10, color=line_color, linestyle='-', linewidth=1)
#plt.axhline(y=v_b, color='r', linestyle='--', label='Spike threshold')
plt.ylabel('Membrane potential (mV)')
plt.xlabel('Time (ms)')
plt.title(f'Membrane Potential of CD Neuron, total frequency: {CD_total_rate:.0f} Hz')
plt.grid()
#plt.savefig("GPe_Membrane_Potential.png")
p_i += 1

#GPe

plt.subplot(p_num, 1, p_i)
for k in samples:
    line, = plt.plot(time_array.cpu().numpy(), v_GPe_values[k].cpu().numpy(), linewidth=1)
    line_color = line.get_color()
    spike_times = time_array[spike_GPe_output[k] == 1].cpu().numpy()  # Get times where spikes occur
    for spike_time in spike_times:
        plt.vlines(x=spike_time, ymin=v_b.cpu().numpy(), ymax=v_b.cpu().numpy() + 10, color=line_color, linestyle='-', linewidth=1)
#plt.axhline(y=v_b, color='r', linestyle='--', label='Spike threshold')
plt.ylabel('Membrane potential (mV)')
plt.xlabel('Time (ms)')
plt.title('Membrane Potential of GPe Neuron with Adjustable Synaptic Strengths and Input Rates')
plt.grid()
#plt.savefig("GPe_Membrane_Potential.png")
p_i += 1

plt.subplot(p_num, 1, p_i)
plt.plot(time_array.cpu().numpy(), torch.mean(I_GPe_values, dim=0).cpu().numpy(), color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Current (nA)')
plt.title('GPe Neuron Mean Synaptic Current')
plt.grid()
#plt.savefig("GPe_Mean_Synaptic_Current.png")
p_i += 1

plt.subplot(p_num, 1, p_i)
plt.plot(I_GPe_freqs[:50].cpu().numpy(), I_GPe_psd[:50].cpu().numpy(), color='purple')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('Power Spectrum Density of GPe Neuron Synaptic Current')
plt.grid()
#plt.savefig("GPe_PSD_Synaptic_Current.png")
p_i += 1

#STN

plt.subplot(p_num, 1, p_i)
for k in samples:
    line, = plt.plot(time_array.cpu().numpy(), v_STN_values[k].cpu().numpy(), linewidth=1)
    line_color = line.get_color()
    spike_times = time_array[spike_STN_output[k] == 1].cpu().numpy()  # Get times where spikes occur
    for spike_time in spike_times:
        plt.vlines(x=spike_time, ymin=v_b.cpu().numpy(), ymax=v_b.cpu().numpy() + 10, color=line_color, linestyle='-', linewidth=1)
#plt.axhline(y=STN.neurons[0].v_b, color='r', linestyle='--', label='Spike threshold')
plt.ylabel('Membrane potential (mV)')
plt.xlabel('Time (ms)')
plt.title('Membrane Potential of STN Neuron with Adjustable Synaptic Strengths and Input Rates')
plt.grid()
#plt.savefig("STN_Membrane_Potential.png")
p_i += 1

plt.subplot(p_num, 1, p_i)
plt.plot(time_array.cpu().numpy(), torch.mean(I_STN_values, dim=0).cpu().numpy(), color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Current (nA)')
plt.title('STN Neuron Mean Synaptic Current')
plt.grid()
#plt.savefig("STN_Mean_Synaptic_Current.png")
p_i += 1

plt.subplot(p_num, 1, p_i)
plt.plot(I_STN_freqs[:50].cpu().numpy(), I_STN_psd[:50].cpu().numpy(), color='purple')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('Power Spectrum Density of STN Neuron Synaptic Current')
plt.grid()
#plt.savefig("STN_PSD_Synaptic_Current.png")

# Adjust the spacing between the subplots
plt.subplots_adjust(hspace=0.5)

# Record the end time
end_time = time.time()

# Calculate and print the duration
elapsed_time = int(end_time - start_time)

# Generate a filename with the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
f_name = f"BG_Simulation of_Scale_{num_scale}_{timestamp}_{elapsed_time}.png"

#Show or Save figure
# plt.show()
plt.savefig(os.path.join(output_dir, f_name), dpi=300, bbox_inches='tight')
plt.close()