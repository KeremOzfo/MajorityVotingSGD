import numpy as np
import matplotlib.pyplot as plt
from os import *
import math
from itertools import cycle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

def special_adress():
    adress=[]
    adress_loss = []
    labels = ['\u03B1=0.85', '\u03B1=0.90', '\u03B1=0.95','\u03B1=1']


    adress.append('Results/85/acc')
    adress.append('Results/90/acc')
    adress.append('Results/95/acc')
    adress.append('Results/100/acc')

    adress_loss.append('Results/85/loss')
    adress_loss.append('Results/90/loss')
    adress_loss.append('Results/95/loss')
    adress_loss.append('Results/100/loss')
    return adress,adress_loss,labels

def compile_results(adress):
    results = None
    f_results = []
    for i, dir in enumerate(listdir(adress)):
        vec = np.load(adress + '/'+dir)
        final_result = vec[len(vec)-1]
        f_results.append(final_result)
        if i==0:
            results = vec/len(listdir(adress))
        else:
           results += vec/len(listdir(adress))
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
    plt.axis([0, 300, 85, 95])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.title('Majority Voting')
    plt.legend()
    plt.grid(True)
    plt.show()

def graph_loss(data, legends, interval):
    marker = ['s', 'v', '+', 'o', '*']
    linestyle = ['-', '--', '-.', ':']
    linecycler = cycle(linestyle)
    markercycler = cycle(marker)
    for d, legend in zip(data, legends):
        x_axis = []
        l = next(linecycler)
        m = next(markercycler)
        for i in range(0, len(d)):
            x_axis.append(i * interval)
        plt.plot(x_axis, d, marker=m, linestyle=l, markersize=2, label=legend)
    plt.axis([0, 300, 0, 2])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
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
labels = special_adress()[2]
results = concateresults(special_adress()[0])
results_loss = concateresults(special_adress()[1])
print(results_loss)
#results = concateresults(locations)
graph(results,labels,intervels)
graph_loss(results_loss,labels,intervels)
#data,legends = compile_results(loc)
#graph(data,labels,intervels)