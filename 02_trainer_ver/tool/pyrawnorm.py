import numpy
import sys

MAX_SHORT = 32767 # -2^15 ~ 2^15-1

# const_energy: baseline constant for energy
# silence: silence threshold
def energy_norm(data, const_energy=None, silence=10000):
    if const_energy is None:
        const_energy = 1500
    
    energy = 0
    count = 0
    # calc energy (sum of squares x(t)*x(t))
    for x in data:
        tmp = x * x
        if tmp > silence:
            energy += tmp
            count += 1
    
    amp = const_energy * numpy.sqrt(count/energy)
    
    # check output max less than max value of short type
    outmax = 0
    for x in data:
        outabs = abs(int(amp * x))
        if outabs > outmax:
            outmax = outabs
    if outmax > MAX_SHORT:
        amp *= MAX_SHORT / outmax
    
    norm_data = amp * data
    return norm_data

def calc_mean_energy(data, silence=10000):
    energy = 0
    count = 0
    # calc energy (sum of squares x(t)*x(t))
    for x in data:
        tmp = data * data
        if tmp > silence:
            energy += tmp
            count += 1
    return int(energy / count)

def max_amplitude_norm(data, max_value=10000):
    _max = data.max()
    
    norm_data = data * max_value / _max
    return norm_data.astype(numpy.int16)
