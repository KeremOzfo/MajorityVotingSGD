import numpy as np
import matplotlib.pyplot as plt
from os import *
import math
from itertools import cycle
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def special_adress():
    adress=[]
    adress_loss = []
    labels = ['SSGD-MV-AD-L1', 'SSGD-MV-AD-L2','SSGD-MV-AD-L2-Q','SSGD-TopK']


    adress.append('Results/add_drop_H_1')
    adress.append('Results/add_drop_H_2')
    adress.append('Results/add_drop_H_2-Q')
    # adress.append('Results/vanilla_H_1')
    # adress.append('Results/vanilla_H_2')
    adress.append('Results/topK')

    return adress,labels

def compile_results(adress):
    results = None
    f_results = []
    total_len = len(listdir(adress)) -1
    for i, dir in enumerate(listdir(adress)):
        if dir[0:3] !='sim':
            vec = np.load(adress + '/'+dir)
            final_result = vec[len(vec)-1]
            f_results.append(final_result)
            if i==0:
                results = vec/total_len
            else:
               results += vec/total_len
    avg = np.average(f_results)
    st_dev = np.std(f_results)
    return results, [adress,avg,st_dev]

def cycle_graph_props(colors,markers,linestyles):
    randoms =[]
    randc = np.random.randint(0,len(colors))
    randm = np.random.randint(0,len(markers))
    randl = np.random.randint(0,len(linestyles))
    m = markers[randm]
    c = colors[randc]
    l = linestyles[randl]
    np.delete(colors,randc)
    np.delete(markers,randm)
    np.delete(linestyles,randl)
    print(colors,markers,linestyles)
    return c,m,l


def avgs(sets):
    avgs =[]
    for set in sets:
        avg = np.zeros_like(set[0])
        avgs.append(avg)
    return avgs

def graph(data, legends,interval):
    marker = ['s', 'v', '+', 'o', '*']
    linestyle =['-', '--', '-.', ':']
    linecycler = cycle(linestyle)
    markercycler = cycle(marker)
    for d,legend in zip(data,legends):
        x_axis = []
        l = next(linecycler)
        m = next(markercycler)
        for i in range(0,len(d)):
            x_axis.append(i*interval)
        plt.plot(x_axis,d, marker= m ,linestyle = l ,markersize=2, label=legend)
    #plt.axis([5, 45,70 ,90])
    #plt.axis([145,155,88,92])
    #plt.axis([290, 300, 87, 95])
    #plt.axis([50, 100, 87, 95])
    plt.axis([0, 200, 65, 70])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.title('Majority Voting')
    plt.legend()
    plt.grid(True)
    plt.show()

def graph2(data, legends,interval):
    marker = ['s', 'v', '+', 'o', '*']
    linestyle = ['-', '--', '-.', ':']
    linecycler = cycle(linestyle)
    markercycler = cycle(marker)
    fig, ax = plt.subplots()
    for d, legend in zip(data, legends):
        x_axis = []
        l = next(linecycler)
        m = next(markercycler)
        for i in range(0, len(d)):
            x_axis.append(i * interval)
        ax.plot(x_axis, d, marker=m, linestyle=l, markersize=2, label=legend)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([65, 70])
    ax.grid(True)
    ax.legend()

    axins1 = zoomed_inset_axes(ax, zoom=2, loc=7)

    for d, legend in zip(data, legends):
        x_axis = []
        l = next(linecycler)
        m = next(markercycler)
        for i in range(0, len(d)):
            x_axis.append(i * interval)
        axins1.plot(x_axis, d)
    axins1.grid(True)
    ax.plot()

    # SPECIFY THE LIMITS
    x1, x2, y1, y2 = 190, 200, 68.8, 69.8
    axins1.set_xlim(x1, x2)
    axins1.set_ylim(y1, y2)
    # IF SET TO TRUE, TICKS ALONG
    # THE TWO AXIS WILL BE VISIBLE
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    mark_inset(ax, axins1, loc1=1, loc2=2, fc="none", ec="0.2")
    plt.draw()
    plt.show()


def concateresults(dirsets):
    all_results =[]
    for set in dirsets:
        all_results.append(compile_results(set)[0])
        print(compile_results(set)[1])
    return all_results



loc = 'Results/'
types = ['benchmark','timeCorrelated','topk']
NNs = ['simplecifar']

locations = []
labels =[]
for tpye in types:
    for nn in NNs:
        locations.append(loc + tpye +'/'+nn)
        labels.append(tpye +'--'+ nn)

intervels = 1
labels = special_adress()[1]
results = concateresults(special_adress()[0])
#results = concateresults(locations)
graph2(results,labels,intervels)
#data,legends = compile_results(loc)
#graph(data,labels,intervels)