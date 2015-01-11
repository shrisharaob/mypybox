import numpy as np
import pylab as plt
import os, sys, datetime
import matplotlib.image as mpimg
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs

def ReportFig(*args):
    argc = len(args)
    print argc
    now = datetime.datetime.now()
    filebase = "/homecentral/srao/Documents/cnrs/figures/reports/"
#    filebase = "./"
    filename = "%s.%s.%s.%s%s.html" %((now.day, now.month, now.year, now.hour, now.minute))
    figurebase = filebase + os.path.splitext(filename)[0] + '/'
    title = "report %s.%s.%s"%((now.day, now.month, now.year))
    pageHeading = ''
    figFormat = "png"
    IF_APPEND = False
    IF_FROM_FILE = False
    fig_caption = "..."
    fig_name = "%s.%s.%s.%s"%((now.hour, now.min, now.second, now.microsecond))

    filename, fig_caption, pageHeading, figFormat, title, fig_name = DefaultArgs(args, [filename, '', pageHeading, figFormat, title, fig_name])

    filename = os.path.splitext(filename)[0]
    filename = filename + ".html"
    figurebase = filebase + os.path.splitext(filename)[0] + '/'
    print figurebase
    
    nFigs = len(os.popen("ls %s*.png"%((figurebase, ))).read().split())
    print nFigs
    

    # if(argc > 0):
    #     filename = args[0]
    #     filename = os.path.splitext(filename)[0]
    #     filename = filename + ".html"
    #     figurebase = filebase + os.path.splitext(filename)[0] + '/'
    # if(argc > 1):
    #     fig_caption = args[1]
    # if(argc > 2):
    #     pageHeading = args[2]
    # if(argc > 3):
    #     figFormat = args[3]
    # if(argc > 4):
    #     title = args[4]
    # if(argc > 5):
    #     fig_name = args[5]

    if not os.path.isdir(figurebase):
        os.system('mkdir %s'%((figurebase, )))

    if os.path.isfile(figurebase + filename):
        print "\nappending to existing report", filename
        IF_APPEND = True
        fp = open(figurebase + filename, 'a')
    else :
        fp = open(figurebase + filename, 'w')
    
    print IF_FROM_FILE
    if IF_FROM_FILE:
        fig_filename = sys.argv[1]       
        print "\ncopying file :"
#        sys.stdout.fflush()
        os.system("cp %s %s -v" %((fig_filename, figurebase)))
    else :
        fig_filename = figurebase + fig_name + "." + figFormat
        plt.savefig(fig_filename)

        #img = mpimg.imread(fig_filename)
        
    width = 500
    height = 400
    html_fontsize = 2
    relativeFigPath = "./" + os.path.basename(fig_filename) 
    if not IF_APPEND:
        fp.write("<html> \n <head> \n <title> %s </title> </head> <body> <h2> %s <br></h2>\n" %((title, pageHeading)))
        fp.write("<img src= \"%s\" alt = Figure %s width= %s height = %s>\n" %((relativeFigPath, 0, width, height)))
        fp.write("<br><br><p><font size = \"%s\">  Fig.%s %s </font> </p> \n <br> <hr> <br> \n </body> \n </html>" %((html_fontsize, nFigs + 1, fig_caption, )))
    else :
        fp.write("\n<html> \n <head> <body>")
        fig_filename = figurebase + fig_name + "." + figFormat
        plt.savefig(fig_filename)
        fp.write("<img src= \"%s\" alt = Figure %s width= %s height = %s>\n" %((relativeFigPath, 0, width, height)))
        fp.write("<br><br><p><font size = \"%s\">  Fig.%s %s </font> </p> \n <br> <hr> <br> \n </body> \n </html>" %((html_fontsize, nFigs+1, fig_caption, )))
#        fp.write("<br><br><p> %s </p> \n <br> <hr> <br> \n </body> \n </html>" %((fig_caption, )))

    fp.close()
    
    


if __name__ == "__main__" :
    plt.ioff()
    plt.plot(np.arange(1000))
    filename = 'tst_filename'
    fig_caption = 'fig_caption'
    pageHeading = 'pageHeading'
    figFormat = 'png'
    title = 'title'
    fig_name = 'fig_name'

    ReportFig(filename, fig_caption, pageHeading, figFormat, title, fig_name)
    
                 
