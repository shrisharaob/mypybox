# script to check if fixed connectivity matrix produces similar firing rates neuuron by neuron
import MySQLdb as mysql
import numpy as np
import os, sys
import pylab as plt
sys.path.append("/homecentral/srao/Documents/code/mypybox")
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
import Keyboard as kb
from reportfig import ReportFig
from multiprocessing import Pool
from functools import partial
from Print2Pdf import Print2Pdf

def ComputeFiringRateDiff(dbName, theta0, theta1, simDuration, neuronsList, NI = 10000):
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName, local_infile = 1)
    db.autocommit(True)
    dbCursor = db.cursor()
    discardTime = 3000.0 # discard first 3s
    frDiff = np.zeros((20000, ))
    print frDiff.shape
    fr  = np.zeros(neuronsList.shape)
    fr0 = np.zeros(neuronsList.shape)
    fr1 = np.zeros(neuronsList.shape)
#    print "MEAN RATE I = ", float(dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId > %s AND theta = %s", (10000, theta0))) / (simDuration * 1e-3 * NI)
    print "MEAN RATE I = ", float(dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId > %s AND theta = %s and spkTimes > %s and spkTimes < %s", (10000, theta0, discardTime, simDuration))) / ((simDuration - discardTime) * 1e-3 * NI)
    #kb.keyboard()
    for kk, kNeuron in enumerate(neuronsList):
#        print kNeuron
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s and spkTimes > %s and spkTimes < %s", ((kNeuron, theta0, discardTime, simDuration)))
        if(nSpks > 10):
            fr0[kk] = float(nSpks) / ((simDuration - discardTime) * 1e-3)
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s and spkTimes > %s and spkTimes < %s", ((kNeuron, theta1, discardTime, simDuration)))
        if(nSpks > 10):
            fr1[kk] = float(nSpks) / ((simDuration - discardTime) * 1e-3)
        if(fr0[kk]>0 and fr1[kk] >0):
            frDiff[kk] = (fr0[kk] - fr1[kk] ) #/ (0.5 * (fr0[kk] + fr1[kk]))
            
    frDiff = frDiff[frDiff!=0]
    dbCursor.close()
    db.close()
    return fr0, fr1, frDiff ** 2


def ComputeFiringRateDiff02(dbName, simDuration, neuronsList, NI, thetaPair):
    theta0 = thetaPair[0]
    theta1 = thetaPair[1]
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName, local_infile = 1)
    db.autocommit(True)
    dbCursor = db.cursor()
    discardTime = 3000.0 # discard first 2s
    frDiff = np.zeros((20000, ))
    print frDiff.shape
    fr  = np.zeros(neuronsList.shape)
    fr0 = np.zeros(neuronsList.shape)
    fr1 = np.zeros(neuronsList.shape)
    print "MEAN RATE I = ", float(dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId > %s AND theta = %s", (10000, theta0))) / (simDuration * 1e-3 * NI)
#    print "MEAN RATE I = ", float(dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId > %s AND theta = %s and spkTimes > %s and spkTimes < %s", (10000, theta0, discardTime, simDuration))) / ((simDuration - discardTime) * 1e-3 * NI)
    #kb.keyboard()
    for kk, kNeuron in enumerate(neuronsList):
#        print kNeuron
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s and spkTimes > %s and spkTimes < %s", ((kNeuron, theta0, discardTime, simDuration)))
        if(nSpks > 10):
            fr0[kk] = float(nSpks) / ((simDuration - discardTime) * 1e-3)
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s and spkTimes > %s and spkTimes < %s", ((kNeuron, theta1, discardTime, simDuration)))
        if(nSpks > 10):
            fr1[kk] = float(nSpks) / ((simDuration - discardTime) * 1e-3)
        if(fr0[kk]>0 and fr1[kk] >0):
            frDiff[kk] = (fr0[kk] - fr1[kk] ) #/ (0.5 * (fr0[kk] + fr1[kk]))
            
    frDiff = frDiff[frDiff!=0]
    dbCursor.close()
    db.close()
    print "Compute rate DONE !"
    return fr0, fr1, frDiff ** 2

if __name__ == "__main__":
    #useDb = 'a8t12' #'init_alpha1' #<-- 8ms  #'alpha1_t5'
    useDb = sys.argv[2]
    simDuration = 100000.0
    NE = 10000
    NI = 10000
    neuronsList = np.arange(0, NE+NI, 1)
    #tau = np.array([3.0, 6.0, 8.0, 10.0, 12.0])
 #   tau = np.array([3.0, 24.0])
    tau = np.array([24.0]);
#    alpha =  [0.9]
    alpha =  [0.5]
    obsWin = np.array([10, 20, 50, 75, 100]) * 1e3 # time windows for estimation of firing rate 
    runType = 2;
    IF_COMPUTE_MSDIFF = False
    if(len(sys.argv) > 1):
        runType = int(sys.argv[1])
    # COMPUTE
    if(runType == 0): 
        print "computing"
        if(IF_COMPUTE_MSDIFF):
            print "computing mse for different window lengths"
            mse_frDiff = np.empty((len(obsWin), len(alpha), len(alpha)))
            mse_frDiff[:] = np.nan
            for ii, iObsWin in enumerate(obsWin):
                for mm, mAlpha in enumerate(alpha):
                    print "alpha : ", mAlpha, "ons win = ", iObsWin
                    theta0 = ['%s%s0'%((int(10*x), int(10 * mAlpha))) for x in tau]
                    theta1 = ['%s%s1'%((int(10*x), int(10 * mAlpha))) for x in tau]
                    for ll, lTau in enumerate(tau):
                        print useDb, theta0, theta1
                        df = ComputeFiringRateDiff(useDb, theta0[ll], theta1[ll], iObsWin, neuronsList)
                        mse_frDiff[ii, mm, ll] = np.mean(df[2])
                        
            np.save('./data/mse_fr_diff' + useDb, mse_frDiff)
            kb.keyboard()
            plt.ioff()
            plt.plot(obsWin * 1e-3, np.squeeze(mse_frDiff), 'ko-')
            plt.xlabel('Firing rate estimation window (s)')
            plt.ylabel(r'mean square diff (${Hz}^2$) $\frac{1}{N}\sum_{i=0}^{N-1}(r_i^1 - r_i^2)^2$', fontsize = 20)
            plt.title(r'$\alpha = %s \; \tau = %s$'%((mAlpha, lTau)), fontsize = 20)
            kb.keyboard()
        else :     

            p = Pool(10)
            nidx = []
            for iii in range(10):
                nidx.append(range(iii*1000, (iii+1)*1000))
            for mm, mAlpha in enumerate(alpha):
                print "alpha : ", mAlpha
                theta0 = ['%s%s0'%((int(10*x), int(10 * mAlpha))) for x in tau]
                theta1 = ['%s%s1'%((int(10*x), int(10 * mAlpha))) for x in tau]
                theta0 = [30]
                theta1 = [31]

                for ll, lTau in enumerate(tau):
                    print useDb, theta0, theta1
                    results = p.map(partial(ComputeFiringRateDiff, useDb, theta0[ll], theta1[ll], simDuration), np.array(nidx))
                    kb.keyboard()
                    df = results
#                    df = ComputeFiringRateDiff(useDb, theta0[ll], theta1[ll], simDuration, neuronsList)
                    fr0 = df[0]
                    fr1 = df[1]
                    np.save('./data/fr_init_alpha%s_tau%s'%((mAlpha, lTau)) + useDb, np.array(df))
                    print " alpha %s tau = %s done"%((mAlpha, lTau))
        
#        kb.keyboard()

    if(runType == 1):
        print "computing, multi ...."
        p = Pool(2)
        for mm, mAlpha in enumerate(alpha):
            print "alpha : ", mAlpha
#            thetaPairs = [[30, 31], [60, 61], [80, 81], [100, 101], [120, 121]]
#            thetaPairs = [[30, 31], [240, 241]]
            thetaPairs = [[240, 241]]
            results = p.map(partial(ComputeFiringRateDiff02, useDb,  simDuration, neuronsList, NI), thetaPairs)
            for ll, lTau in enumerate(tau):
                print useDb, thetaPairs[ll]
                df = results[ll]
                fr0 = df[0]
                fr1 = df[1]
                np.save('./data/fr_init_alpha%s_tau%s'%((mAlpha, lTau)) + useDb, np.array(df))
                print " alpha %s tau = %s done"%((mAlpha, lTau))
        
        kb.keyboard()
        


    #### ALPHA = 1 ####
        # print "alpha 1"
        # theta0 = 12110
        # theta1 = 12111
        # df = ComputeFiringRateDiff(useDb, theta0, theta1, simDuration, neuronsList)
        # fr0 = df[0]
        # fr1 = df[1]
        # np.save('fr_init_alpha1_'+useDb, np.array(df))
        # kb.keyboard()
    if(runType == 2):
    # DISPLAY
        print "plotting"
        plt.figure()
        plt.ioff()
        figFolder = "/homecentral/srao/Documents/code/mypybox/nda/figs/"
 
        for mm, mAlpha in enumerate(alpha):
            for ll, lTau in enumerate(tau):
                filename = "initial_condition_alpha%s_tau%s_T%s"%((mAlpha, lTau, simDuration))
                print "alpha = ", mAlpha, " tau = ", lTau
                print './data/fr_init_alpha%s_tau%s'%((mAlpha, lTau)) + useDb
                df = np.load('./data/fr_init_alpha%s_tau%s'%((mAlpha, lTau)) + useDb + '.npy')
                print df.shape
                df = np.array([df[0], df[1]])
                plt.plot(df[0, NE:], df[1, NE:], 'r.', label='I')
                plt.plot(df[0, :NE], df[1, :NE], 'k.', label='E')
                plt.title(r'$\alpha = %s \; \tau = %s$'%((mAlpha, lTau)), fontsize = 20)
                plt.xlabel('firing rates (Hz), condition 1', fontsize = 20)
                plt.ylabel('firing rates (Hz), condition 2', fontsize = 20)
                plt.legend(prop={"size":18})
                plt.tick_params(axis = 'both', labelsize = 16)
                #plt.show()
#                ReportFig('init_cond_ff_alpha%s'%((int(10 * alpha[0]),   )), 'Neuron-wise firing rates for two different initial conditions, with connection matrix fixed, simulation time = %ss'%((simDuration * 1e-3)), 'Dependence on initial condition', 'png', 'initial condition', 'alpha%s_tau%s_T%s'%((mAlpha, lTau, simDuration)))
                
                Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png') #, tickFontsize=14, paperSize = [4.0, 3.0])
                plt.clf()



        # df = np.load('fr_init_alpha1_' + useDb + '.npy')
        # plt.plot(df[0, NE:], df[1, NE:], 'r.', label='I')
        # plt.plot(df[0, :NE], df[1, :NE], 'k.', label='E')
        # plt.title(r'$\alpha = 1$', fontsize = 20)
        # plt.xlabel('firing rates (Hz), condition 1', fontsize = 20)
        # plt.ylabel('firing rates (Hz), condition 2', fontsize = 20)
        # plt.legend(prop={"size":18})
        # plt.tick_params(axis = 'both', labelsize = 16)
        # plt.show()
        # ReportFig('initial_condition', 'Neuron-wise firing rates for two different initial conditions, with connection matrix fixed', 'Dependence on initial condition', 'png', 'initial condition', 'alpha%s_tau%s'%((1, tau)))

        #kb.keyboard()




    # f, ax = plt.subplots(2, 2)
    # cnts, bins, patches = ax[0, 0].hist(df[0:NE], 100, label='E')
    # plt.setp(patches, 'facecolor', 'k')
    # ax[0, 0].set_title(r'$\alpha = 0, \tau = 3ms$', fontsize=20)
    # ax[0, 0].legend()
    # cnts, bins, patches = ax[0, 1].hist(df[NE:], 100, label='I')
    # plt.setp(patches, 'facecolor', 'r', 'edgecolor', 'r')
    # ax[0, 1].set_title(r'$\alpha = 0, \tau = 3ms$', fontsize=20)
    # ax[0, 1].legend()


    # cnts, bins, patches = ax[1, 0].hist(df[0:NE], 100, label='E')
    # plt.setp(patches, 'facecolor', 'k')
    # ax[1, 0].set_title(r'$\alpha = 1, \tau = 3ms$', fontsize=20)
    # ax[1, 0].set_xlabel('neuron-wise difference of mean firing rates', fontsize = 19)
    # ax[1, 0].set_ylabel('count', fontsize = 20)
    # cnts, bins, patches = ax[1, 1].hist(df[NE:], 100, label='I')
    # plt.setp(patches, 'facecolor', 'r')
    # plt.setp(patches, 'facecolor', 'r', 'edgecolor', 'r')
    # ax[1, 1].set_title(r'$\alpha = 1, \tau = 3ms$', fontsize=20)
    # plt.show()
    # plt.savefig('neuron_wise_fr_diff')
