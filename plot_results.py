#!/usr/bin/env python

'''
This file plots training convergence data.

Created on 4/4/2023 by Steven Laverty (lavers@rpi.edu)
'''

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    single_data = -np.load('single-seq.npy')
    superfam_data = -np.load('single-superfam.npy')
    plt.rc('font', size=28)
    plt.figure(figsize=(14.4, 10))
    plt.plot(np.arange(len(single_data)), single_data, label='Single sequence:\nIGG-Binding Protein\n(Streptococcus G148)')
    plt.plot(np.arange(len(superfam_data)), superfam_data, label='255 sequences:\nCATH superfamily\n3.10.20.10')
    plt.yscale('log')
    plt.yticks(np.logspace(0.6, 1.7, 5), ['{:.1f}'.format(x) for x in np.logspace(0.5, 1.5, 5)])
    plt.xlabel('Number of training iterations')
    plt.ylabel('Average FAPE (log scale)')
    plt.title('Deep Q learning convergence')
    plt.legend()
    plt.savefig('convergence.png')

    print('Single sequence minimum', np.min(single_data), np.argmin(single_data))
    print('Single superfamily minimum', np.min(superfam_data), np.argmin(superfam_data))
