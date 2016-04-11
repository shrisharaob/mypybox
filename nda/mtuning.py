#this script stiches together the fano factor calculation form two different simulations at different simuli orientations

basefolder = '/homecentral/srao/Documents/code/mypybox/nda/spkStats/data/'
ff0 = np.load(basefolder + 'FFvsOri_gx175.npy')
ff1 = np.load(basefolder + 'FFvsOri_fftuningDelThis.npy')
tc0 = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_gx175.npy')
tc1 = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_fftuningDelThis.npy')

circVar = np.load('/homecentral/srao/Documents/code/mypybox/db/data/Selectivity_gx175.npy')


