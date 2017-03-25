import numpy as np
import astropy.units as u
import subprocess
from tempfile import TemporaryFile		# Enables file-saving/loading.
import matplotlib.pyplot as plt
import scipy.linalg as sla

# 1.29.17 - Manipulating and displaying USB radio antenna data. 

print('"SpectrumPlot.py" \n \nAvailable functions: \n  readradio - Reads the data from "sampler"\'s output.\n  savecustom - Saves a custom preset for frequency, sample frequency, etc.\n  loadcustom - Loads a custom preset.\n  specplot - Plots the spectrum vs time of a given frequency\'s data file.\n  disperser - Add description here. \n  addnoise - Add description here. \n')


def readradio(fname='radioline.bin',Nchannel=5000,frequency = 96.7*u.MHz, rate = 3*u.MHz, timesamples=False,\
             mode='FFT'):
    '''
    Reads the recorded data files of a certain frequency, and then outputs
    arrays of time-averaged 'spectrum' (via FFT or KLT), the spectrum as a function
    of time, and the corresponding frequencies and time.
    
    Parameters:
    -----------
    fname : str
        Name of the data file to be read, including
        '.bin' suffix.
    Nchannel : int
        Number of samples.
    frequency : float
        Frequency of sampled radio line, in MHz.
    rate : float
        Sampling rate, in MHz (millions of
        samples per second).
    timesamples : bool
        Returns the raw array if True.
        Default: False
    mode : str ('FFT' or 'KLT')
        Analyzes data using fast fourier transform or
        Karhunen-Loeve transform.
        Default: 'FFT'
       
    Returns:
    --------
    if mode=='FFT':
        fftd_ave : array
            1D array of time-averaged spectrum
            vs frequency.
        freq : array
            1D array of corresponding frequencies to
            fftd_ave and fftd.
        fftd : array
            2D array of _complex_ power vs frequency,
            as a function of time.
            OR
            2D array of eigenvalues vs eigenvalue
            number, as a function of time.
        time : array
            1D array of corresponding time.
    elif mode=='KLT':
        eigval : array
            2D array of eigenvalues vs eigenvalue
            number, as a function of time.
        #eigval_n : array
        #    2D array of eigenvalues vs eigenvalue
        #    number, as a function of time, for
        #    NOISE (as opposed to measured signal).
        time : array
            1D array of corresponding time.
    '''
    d = np.fromfile(fname,dtype=np.uint8)
    d = d.astype(np.float)
    d -= 127
    print d.shape
    d = d[0::2]+d[1::2]*1j
    if timesamples:
        return(d)
    d.shape = (d.size/Nchannel,Nchannel) # Navgs by Nchannels
    

    if mode=='fft' or mode=='FFT':
        # Fast Fourier Transform
        fftd = np.fft.fft(d,axis=1)           # Spectrum
        fftd_ave = np.mean(np.abs(fftd),axis=0)   # Time-averaged spectrum vs frequency
        
        
        freq = np.fft.fftfreq(Nchannel)*rate+frequency
        print d.size
        time = (np.arange(d.size/Nchannel)/rate)

        freq = np.roll(freq, np.size(freq)/2)         # Array of frequencies, fixed to range from lowest to highest.
        fftd_ave = np.roll(fftd_ave, np.size(fftd_ave)/2)# Array of corresponding average spectrum, fixed the same way.
        fftd = np.roll(fftd, np.size(freq)/2, axis=1) # Array of corresponding spectrum AT DIFFERENT TIMES, also fixed.
        
        return fftd_ave,freq,fftd,time
        
    elif mode=='klt' or mode=='KLT':
        folds = 10       # (!) Change this manually if you want to run it faster or slower. Recommended: 5.
        step = 10        # (!) Change this manually if you want to run it faster, at the cost of imshow resolution.
                         #        Default: 1.
        scale = 1        # (!) Affects the rms of the noise. Leave at 1 unless otherwise indicated.
        
        # Let's reshape the signal into  
        d.shape = (d.shape[0]*folds, d.shape[1]/folds)
        
        
        # KL Transform: SIGNAL
        ffty = np.fft.fft(d, axis=0)
        acorfft = np.fft.ifft(ffty * np.conj(ffty), axis=0)
        # Drop the imaginary component.
        acorfft = acorfft.real
        # Magic of the KL Transform is just to calculate the eigenvalues of the Toeplitz matrix.
        print acorfft.size
        print acorfft.shape
        eigval = np.zeros(acorfft.size/step).reshape(acorfft.shape[0]/step,acorfft.shape[1]) 
                 # Creates empty array of same size as acorfft, but with shape altered by number of stats.
        
        print 'acorfft.shape ='+str(acorfft.shape)
        for i in range(0,acorfft.shape[0],step):
            toeplitz_matrix = sla.toeplitz(acorfft[i])
            #print i
            eigval[i/step], dummy = np.linalg.eigh(toeplitz_matrix)  # Don't bother with eigenvectors.

        time = (np.arange(d.size/Nchannel)/rate)
        
        # KL Transform: NOISE
        #eigval_n = addnoise_KLT_r(d,eigval,step,scale)
        
        #return eigval,eigval_n,time
        return eigval,time
    
    else:
        print "ERROR : 'mode' must equal FFT' or 'KLT'."
        return


def savecustom(frequency, Nchannel, rate):
    # When saving, all values must be UNITLESS. Units will be applied in other functions.
    
    custom_index=['frequency','Nchannel','rate']     # Frequency of radio signal, number of samples, sampling rate.
    custom = [frequency, int(Nchannel), rate]
    
    outputsuffix = str(frequency).replace(".","_")     # e.g. if freq = 96.5, then outputsuffix = '96_5'.
    outfile = "config_"+outputsuffix+".bin"
    print outfile
    f = file(str(outfile),"wb")                  # Saves the following into 'filename.bin'.
    np.save(f,custom)                      # Saves array1.
    np.save(f,custom_index)                # Saves array2.
    f.close()                                    # No more saving; back to regular code.
    
def loadcustom(frequency):
    # All output will be UNITLESS. Units will be applied in other functions.

    outputsuffix = str(frequency).replace(".","_")     # e.g. if freq = 96.5, then outputsuffix = '96_5'.
    outfile = "config_"+outputsuffix+".bin"

    f = file(str(outfile),"rb")                 # Loads 'filename.bin'.
    custom = np.load(f)                         # Loads the old 'array1' into ARRAY1.
    custom_index = np.load(f)               # Loads the old 'array2' into ARRAY2. Note that this is 
                                            #     based on the order in which the arrays were saved! 
                                            #     You don't get to specify which array to load
                                            #     based on its name.
    f.close()                                   # No more loading; back to regular code.

    return custom[0],int(custom[1]),custom[2]
#    return custom[np.where(custom_index=='frequency')[0][0]], custom[np.where(custom_index=='Nchannel')[0][0]], \
#            custom[np.where(custom_index=='rate')[0][0]]    # Keep this here in case we want to add more values
                                                            # to the 'custom' files or call them separately.

def generate(frequency, Nchannel=5000, rate=3, preset=True):
    '''
    Transforms input signal using FFT *and* KLT, then saves 
    the resulting arrays. It can be called on elsewhere in
    the code for convenience.

    Parameters:
    -----------
    frequency : float
        Frequency of sampled radio line, in MHz.
    Nchannel : int
        Number of samples.
    rate : float
        Sampling rate, in MHz (millions of
        samples per second).
    preset : bool
        Toggles whether a preset is activated.
        Default: True
        (!) False is currently unsupported.
        
    Returns:
    --------
    None
    '''
    
    if preset==True:
        frequency,Nchannel,rate = loadcustom(frequency) # Radio freq. in MHz; No. of samples; sampling freq. in MHz.
    outputsuffix = str(frequency).replace(".","_")      # e.g. if freq = 96.5, then outputsuffix = '96_5'.

        
    # Generates FFT array:
    spec, freq, fftd, time = readradio(frequency = frequency*u.MHz, Nchannel=Nchannel, \
                                  rate=rate*u.MHz, fname = "radioline_"+outputsuffix+".bin",\
                                  mode='FFT')
    print "Frequency = "+str(frequency)+" MHz: FFT Complete"
    
    # Generates KLT array, and makes it presentable:
    eigval, time = readradio(frequency = frequency*u.MHz, Nchannel=Nchannel, \
                                rate=rate*u.MHz, fname = "radioline_"+outputsuffix+".bin",\
                                mode='KLT')
    eigval = np.abs(eigval)     # Takes absolute value of eigenvalues.
    eigval = np.sort(eigval)    # Lists |eigval| in ascending order.
    eigval = eigval[:,::-1]     # Reverses the order of |eigval|, so that they descend rather than ascend.
    print "Frequency = "+str(frequency)+" MHz: KLT Complete"
    
    
    
    # Saving FFT and KLT arrays
    outfileFFT = "save_FFT_"+outputsuffix+".bin"
    outfileKLT = "save_KLT_"+outputsuffix+".bin"
    #outfileTIME = "save_time_"+outputsuffix+".bin"
    
    #print type(fftd)
    #print type(eigval)
    #print type(time)
    
    f1 = file(str(outfileFFT),"wb")                  # Saves the following into 'filename.bin'.
    np.save(f1,fftd) 
    f1.close()
    
    f2 = file(str(outfileKLT),"wb")
    np.save(f2,eigval)
    f2.close()
    
    #f3 = file(str(outfileTIME),"wb")
    #np.save(f3,time2)
    #f3.close()
    
    return


def specplot(frequency, Nchannel=5000, rate=3, preset=True, mode='FFT', noisemode=1):
    '''
    Plots spectrum or eigenvalues versus time from a given data file, then
    saves it.

    Parameters:
    -----------
    frequency : float
        Frequency of sampled radio line, in MHz.
    mode : str ('FFT' or 'KLT')
        Analyzes data using fast fourier transform or
        Karhunen-Loeve transform.
    noisemode : int
        Controls how noise affects output, for both FFT and KLT.
        Note that noisemodes 4 and up are KLT only.
        noisemode == 0: Returns an error.
        noisemode == 1: Plots signal only. (DEFAULT)
        noisemode == 2: Separate plots for signal and noise.
        noisemode == 3: Single plot of "signal-noise".
        noisemode == 4: Single plot of signal and noise, for t=0.
        noisemode == 5: Single plot of signal and noise, averaged.
        
    Returns:
    --------
    if mode=='FFT':
        spectrum : array
            2D array of log of spectrum (log(abs(power))
            vs frequency) as a function of time.
            Basically, just the log of fftd.
        spectrum_n : array
            2D array of log of spectrum (log(abs(power))
            vs frequency) as a function of time, for noise.
            Basically, just the log of fftd_n.
        f_min : float
            Minimum value for frequency
            in the 'spectrum'/'fftd' array.
        fmax : float
            Maximum value for frequency
            in the 'spectrum'/'fftd' array.
        tmax : float
            Time elapsed for sampling the data.
    elif mode=='KLT':
        eigval : array
            2D array of eigenvalues vs. eigenvalue
            number as a function of time.
        eigval_n : array
            Same as eigval, but for noise instead
            of signal.
        tmax : float
            Time elapsed for sampling the data.
    '''

    # (!) Change these parameters.
    frequency_n = 1420.0
    Nmax = 100       # Number of eigenvalues to be displayed on x-axis.
    
    frequency,Nchannel,rate = loadcustom(frequency) # Radio freq. in MHz; No. of samples; sampling freq. in MHz.
    outputsuffix = str(frequency).replace(".","_")      # e.g. if freq = 96.5, then outputsuffix = '96_5'.
    outputsuffix_n =str(frequency_n).replace(".","_")      # Same as above, but for noise.
    
    
    # LOADING all data:
    outfileFFT = "save_FFT_"+outputsuffix+".bin"
    outfileKLT = "save_KLT_"+outputsuffix+".bin"

    f1 = file(str(outfileFFT),"rb")                 # Loads saved FFT array.
    fftd = np.load(f1)
    f1.close()

    f2 = file(str(outfileKLT),"rb")                 # Loads saved KLT array.
    eigval = np.load(f2)
    f2.close()
    
    # LOADING all data for noise:
    outfileFFT_n = "save_FFT_"+outputsuffix_n+".bin"
    outfileKLT_n = "save_KLT_"+outputsuffix_n+".bin"

    f3 = file(str(outfileFFT_n),"rb")                 # Loads saved FFT array for noise.
    fftd_n = np.load(f3)
    f3.close()

    f4 = file(str(outfileKLT_n),"rb")                 # Loads saved KLT array for noise.
    eigval_n = np.load(f4)
    f4.close()

    
    if noisemode==0:
        print "ERROR: Noisemode must be an integer larger than 0."
        return
    
    if mode=='fft' or mode=='FFT':
        freq = np.fft.fftfreq(Nchannel)*rate+frequency
        freq = np.roll(freq, np.size(freq)/2)
        
        # PLOTTING: Spectrum (intensity vs frequency) vs Time

        dt = (Nchannel/(rate*1e6)) # Size of time step, in seconds.

        spectrum = np.log(np.abs(fftd))
        spectrum_n = np.log(np.abs(fftd_n))
        fmin = freq[0]
        fmax = freq[-1]
        tmin = 0
        tmax = dt * spectrum.shape[0]


        xmin=fmin
        xmax=fmax
        ymin=0
        ymax= tmax

        # Wrong Noisemode
        if noisemode >= 4:
            print "ERROR: Noisemode must be 1, 2, or 3 for FFTs."
            return spectrum,fmin,fmax,tmax
        
        # Draw Spectrum vs. Time for SIGNAL
        if noisemode==1 or noisemode==2:
            plt.imshow(spectrum, extent = [xmin,xmax,ymin,ymax], aspect='auto', \
                       vmin = np.min(spectrum), vmax = np.max(spectrum), origin='lower')
            plt.colorbar()
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Time (s)")
            plt.title("Spectrum vs Time, for "+str(frequency)+" MHz")

            plt.savefig('spec_vs_time_'+outputsuffix+'_MHz.png')
            plt.clf()
        
        # Draw Spectrum vs. Time for NOISE
        if noisemode==2:
            plt.imshow(spectrum_n, extent = [xmin,xmax,ymin,ymax], aspect='auto', \
                       vmin = np.min(spectrum), vmax = np.max(spectrum), origin='lower')
            plt.colorbar()
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Time (s)")
            plt.title("Spectrum vs Time for NOISE, compared to "+str(frequency)+" MHz")

            plt.savefig('spec_vs_time_'+outputsuffix+'_MHz_n.png')
            plt.clf()
        
        # Draw Spectrum vs. Time for SIGNAL-NOISE
        if noisemode==3:
            plt.imshow(spectrum - spectrum_n, extent = [xmin,xmax,ymin,ymax], aspect='auto', \
                       vmin = np.min(spectrum), vmax = np.max(spectrum), origin='lower')
                       #origin='lower')      # In case the above min/max values cause the image to be unclear.
            plt.colorbar()
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Time (s)")
            plt.title("Spectrum vs Time for SIGNAL - NOISE, for "+str(frequency)+" MHz")

            plt.savefig('spec_vs_time_'+outputsuffix+'_MHz_signalonly.png')
            plt.clf() 
        #return spectrum,fftd,fmin,fmax,tmax
        return spectrum,fmin,fmax,tmax
    
    elif mode=='klt' or mode=='KLT':        
        
        # PLOTTING: Eigenvalues (eigval vs eigval number) vs Time

        dt = (Nchannel/(rate*1e6)) # Size of time step, in seconds.
        
        tmin = 0
        tmax = dt * eigval.shape[0]
        ymin=0
        ymax= tmax

        #eigval = np.abs(eigval)     # Takes absolute value of eigenvalues.
        #eigval = np.sort(eigval)    # Lists |eigval| in ascending order.
        #eigval = eigval[:,::-1]     # Reverses the order of |eigval|, so that they descend rather than ascend.
        vmax = np.average(eigval)*10     # Maximum value that will be registered on the color bar.
        
        if noisemode!=1:
            #eigval_n = np.abs(eigval_n)  # Repeat for eigval_n.
            #eigval_n = np.sort(eigval_n) #
            #eigval_n = eigval_n[:,::-1]  #  
            vmax_n = np.average(eigval_n)*10 # Same as vmax, but for noise.
        
        # Plotting & saving, for signal.
        if noisemode==1 or noisemode==2:
            plt.imshow(eigval[:,:Nmax], extent = [0,Nmax,ymin,ymax], \
                       vmin=0, vmax=vmax, aspect='auto', origin='lower')
            plt.xlabel("Eigenvalue Number")
            plt.ylabel("Time (s)")
            plt.title("Eigenvalues vs Time, for "+str(frequency)+" MHz")
            plt.colorbar()

            plt.savefig('eig_vs_time_'+outputsuffix+'_MHz.png')
            plt.clf()
        
        # Plotting & saving, for noise.
        if noisemode==2:
            plt.imshow(eigval_n[:,:Nmax], extent = [0,Nmax,ymin,ymax], \
                       vmin=0, vmax=vmax_n, aspect='auto', origin='lower')
            plt.xlabel("Eigenvalue Number")
            plt.ylabel("Time (s)")
            plt.title("Eigenvalues vs Time ("+str(frequency)+" MHz), for NOISE ONLY.")
            plt.colorbar()

            plt.savefig('eig_vs_time_'+outputsuffix+'_MHz_n.png')
            plt.clf()
        
        # Plotting & saving, for signal-noise.
        if noisemode==3:
            plt.imshow(eigval[:,:Nmax] - eigval_n[:,:Nmax], extent = [0,Nmax,ymin,ymax], \
                       vmin=0, vmax=vmax, aspect='auto', origin='lower')
            plt.xlabel("Eigenvalue Number")
            plt.ylabel("Time (s)")
            plt.title("Eigenvalues vs Time for "+str(frequency)+" MHz, for SIGNAL - NOISE.")
            plt.colorbar()

            plt.savefig('eig_vs_time_'+outputsuffix+'_MHz_signalonly.png')
            plt.clf()        
        
        # Plotting and saving, for time=0 of both signal and noise.
        if noisemode==4:
            axes = plt.gca()
            plt.plot(eigval[:,:Nmax][0], label='Signal')
            plt.plot(eigval_n[:,:Nmax][0], label='Noise')
            axes.set_ylim(0,vmax)
            plt.xlabel("Eigenvalue Number")
            plt.ylabel("Eigenvalue")
            plt.title("Eigenvalues for "+str(frequency)+" MHz, at time=0")
            plt.legend(loc='upper right')
            
            plt.savefig('eig_'+outputsuffix+'_MHz_zerotime')
            plt.clf()
        
        # Plotting and saving, for time-average of both signal and noise.
        if noisemode==5:
            axes = plt.gca()
            plt.plot(np.average(eigval,axis=0)[:Nmax], label='Signal')
            plt.plot(np.average(eigval_n,axis=0)[:Nmax], label='Noise')
            axes.set_ylim(0,vmax)
            plt.xlabel("Eigenvalue Number")
            plt.ylabel("Eigenvalue")
            plt.title("Eigenvalues for "+str(frequency)+" MHz, time-averaged")
            plt.legend(loc='upper right')
            
            plt.savefig('eig_'+outputsuffix+'_MHz_timeaveraged')
            plt.clf()
        
        return eigval, eigval_n, tmax
    else:
        print "ERROR : 'mode' must equal FFT' or 'KLT'."
        return
    

def addnoise(frequency,mode,scale=1.20):
    '''
    Generates array of _complex_ power vs
    frequency, then adds recorded noise with a
    scale factor of 'scale' to it.

    Parameters:
    -----------
    frequency : float
        Frequency of sampled radio line, in MHz.
    scale : float (default=1.20)
        Multiplier for standard deviation of noise.
        e.g. scale==2 will cause noise to have a 
        standard deviation of 2.
    mode : str ('FFT' or 'KLT')
        Analyzes data using fast fourier transform or
        Karhunen-Loeve transform.
        (?) KLT is currently not properly supported.
        
        
    Returns:
    --------
    spectrum_sum : array
        2D array of log of spectrum (log(abs(power))
        vs frequency) + scale*(log of noise spectrum),
        as a function of time.
    eigval_sum : array
        2D array of (signal eigenvalues + noise
        eigenvalues) vs eigenvalue
        number, as a function of time.
        (?) Is this correct?
    '''
    

    frequency_n = 1420.0                   # Frequency of noise to be compared to signal.
    frequency_n_added = 1420.000001        # Frequency of noise to be added directly to signal.
    Nmax = 100                      # Number of eigenvalues to be displayed on x-axis.
    
    frequency,Nchannel,rate = loadcustom(frequency) # Radio freq. in MHz; No. of samples; sampling freq. in MHz.
    outputsuffix = str(frequency).replace(".","_")               # e.g. if freq = 96.5, then outputsuffix = '96_5'.
    outputsuffix_n =str(frequency_n).replace(".","_")                  # Same as above, but for noise.
    outputsuffix_n_added =str(frequency_n_added).replace(".","_")      # Same as above, but for noise_added.
    
    
    
    # -------- LOADING ---------
    # LOADING all data for signal:
    outfileFFT = "save_FFT_"+outputsuffix+".bin"
    outfileKLT = "save_KLT_"+outputsuffix+".bin"

    f1 = file(str(outfileFFT),"rb")                 # Loads saved FFT array.
    fftd = np.load(f1)
    f1.close()

    f2 = file(str(outfileKLT),"rb")                 # Loads saved KLT array.
    eigval = np.load(f2)
    f2.close()
    
    # LOADING all data for noise:
    outfileFFT_n = "save_FFT_"+outputsuffix_n+".bin"
    outfileKLT_n = "save_KLT_"+outputsuffix_n+".bin"

    f3 = file(str(outfileFFT_n),"rb")                 # Loads saved FFT array for noise.
    fftd_n = np.load(f3)
    f3.close()

    f4 = file(str(outfileKLT_n),"rb")                 # Loads saved KLT array for noise.
    eigval_n = np.load(f4)
    f4.close()
    
    # LOADING all data for noise_added:
    outfileFFT_n_added = "save_FFT_"+outputsuffix_n_added+".bin"
    outfileKLT_n_added = "save_KLT_"+outputsuffix_n_added+".bin"

    f3 = file(str(outfileFFT_n_added),"rb")                 # Loads saved FFT array for noise_added.
    fftd_n_added = np.load(f3)
    f3.close()

    f4 = file(str(outfileKLT_n_added),"rb")                 # Loads saved KLT array for noise_added.
    eigval_n_added = np.load(f4)
    f4.close()
    # ------- LOADING COMPLETE -------
    
    
    # Loading is done, and we have fftd and eigval arrays for signal, noise (compared), and noise (added).
    
    
    # ADDING NOISE:
    spectrum = np.log(np.abs(fftd))
    spectrum_n = np.log(np.abs(fftd_n))
    spectrum_n_added = np.log(np.abs(fftd_n))
    
    spectrum_n = (scale+1)*spectrum_n          # This is so that we won't always have Psig > Pnoise when comparing.
    spectrum_n_added = scale*spectrum_n_added
    eigval_n_added = scale*eigval_n_added
    
    
    FFTnormalizingmaximum = np.max(spectrum)
    KLTnormalizingmaximum = np.max(eigval)
    
    
    spectrum_sum = spectrum + spectrum_n_added # This is (Spectrum + Scale*Noise). We're focusing on this.
    eigval_sum = eigval + eigval_n_added       # Same as above, but for eigval. (?) Is this correct, though?

    spectrum_sum = spectrum_sum / np.max(spectrum_sum) * FFTnormalizingmaximum
    eigval_sum = eigval_sum / np.max(eigval_sum) * KLTnormalizingmaximum
    
    # PLOTTING:
    dt = (Nchannel/(rate*1e6)) # Size of time step, in seconds.
    freq = np.fft.fftfreq(Nchannel)*rate+frequency
    freq = np.roll(freq, np.size(freq)/2)

    fmin = freq[0]
    fmax = freq[-1]
    tmin = 0
    tmax = dt * spectrum.shape[0]

    xmin=fmin
    xmax=fmax
    ymin=0
    ymax= tmax
    
    if mode=='FFT' or mode=='fft':
        plt.imshow(spectrum_sum, extent = [xmin,xmax,ymin,ymax], aspect='auto', \
                   vmin = np.min(spectrum_sum), vmax = np.max(spectrum_sum), origin='lower')
        plt.colorbar()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (s)")
        plt.title("Spectrum vs Time ("+str(frequency)+" MHz), for (Signal+"+str(scale)+"*Noise)")

#        plt.savefig('spec_vs_time_'+outputsuffix+'_MHz_signalplusnoise.png')
#        plt.clf()

        # COMPARE: Is there a signal?
        #print "Power sum      for " + str(frequency)+' MHz: ' + str(np.sum(spectrum_sum**2))
        #print "Noisepower sum for " + str(frequency)+' MHz: ' + str(np.sum(spectrum_n**2))
        #print "Sig:Noi Ratio  for " + str(frequency)+' MHz: ' + str(np.sum(spectrum_sum**2) / np.sum(spectrum_n**2))
        #print "\n"+'Is there signal? -> '+str(np.sum(spectrum_sum**2) / np.sum(spectrum_n**2) > 1.00)
        # (!) We're doing it by eye for now. Uncomment these and edit them once we move on to a computational method.

    elif mode=='KLT' or mode=='klt':
        vmax = np.average(eigval)*10     # Maximum value that will be registered on the color bar.
        
        plt.imshow(eigval_sum[:,:Nmax], extent = [0,Nmax,ymin,ymax], \
                   vmin=0, vmax=vmax, aspect='auto', origin='lower')
        plt.xlabel("Eigenvalue Number")
        plt.ylabel("Time (s)")
        plt.title("Eigenvalues vs Time ("+str(frequency)+" MHz), for (Signal+"+str(scale)+"*Noise)")
        plt.colorbar()

#        plt.savefig('eig_vs_time_'+outputsuffix+'_MHz_signalplusnoise.png')
#        plt.clf()      

    
    return spectrum_sum, eigval_sum
    
    
def disperser(frequency,spectrum,fmin,fmax,tmax,DM):
    '''
    Shifts the elapsed time of a plot of spectrum
    vs time, as if it had passed through a plasma of 
    dispersion measure 'DM'.
    
    Parameters:
    -----------
    frequency : float
        Frequency of the sampled radio line,
        in MHz.
    spectrum : array
        2D array of log of spectrum (log(abs(power))
        vs frequency) as a function of time.
    f_min : float
        Minimum and maximum values for frequency
        in the 'spectrum' array.
    fmax : float
        Minimum and maximum values for frequency
        in the 'spectrum' array.
    tmax : float
        Time elapsed for sampling the data, without
        DM taken into account.
    DM : float
        Dispersion measure of the hypothetical
        cloud of gas, in CGS units (cm^-2).
    
    Returns:
    --------
    tau : array
        The time delay from passing the signal
        through a cloud of dispersion measure DM,
        corresponding with frequency range.

    '''
    
    outputsuffix = str(frequency).replace(".","_")     # e.g. if freq = 96.5, then outputsuffix = '96_5'.

    q_e = 4.84e-10     # Elementary charge, in esu
    c_l = 3.0e10        # Speed of light, in cm/s
    m_e = 9.109e-28     # Mass of electron, in g
    
    frange = np.linspace(fmin,fmax,spectrum.shape[1])    # Array containing the range of frequencies.
    tstep = tmax / spectrum.shape[0]                     # Time per "step" in array, in seconds
    
    tau = q_e**2 / (2 * np.pi * c_l * m_e * (frange*1e6)**2) * DM * 3.086e16
                                                                     # Time delayed by the dispersion measure DM
    # tau = np.zeros(spectrum.shape[1]) + np.random.rand(spectrum.shape[1])*0.5   # Testing the code for larger 'tau'.
    
    
    # Expanding Array by As Large as Necessary
    
    n_rows = np.int(np.max(tau) / tstep)        # Number of extra rows the array needs
    blankspace = np.zeros(spectrum.shape[1]*n_rows).reshape(n_rows,spectrum.shape[1])
    spectrum = np.vstack([spectrum,blankspace])
    
    # SHIFTING the Array by Tau
    tau_step = tau / tstep              # tau, in units of steps
    tau_step = tau_step.astype(int)
    
    for i in range(0,spectrum.shape[1]):
        spectrum[:,i] = np.roll(spectrum[:,i],tau_step[i])
    
    
    # PLOTTING: Spectrum (intensity vs frequency) vs Time    
    
    xmin=fmin
    xmax=fmax
    ymin=0
    ymax= tmax + np.max(tau)

    plt.imshow(spectrum, extent = [xmin,xmax,ymin,ymax], aspect='auto', origin='lower')
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (s)")
    plt.title("Spectrum vs Time")

    plt.savefig('spec_vs_time_'+outputsuffix+'_MHz_DM_is_'+str(DM)+'.png')
    
    return tau


def run(FFTmode=0,KLTmode=0):
    '''
    Generates plots of KLTs and FFTs for many recorded
    signals at once via a simple loop.
    
    Parameters:
    -----------
    FFTmode : int
        Controls how noise affects output, for FFT.
        FFTmode == 0: Does not compute FFTs. (DEFAULT)
        FFTmode == 1: Plots signal only.
        FFTmode == 2: Separate plots for signal and noise.
        FFTmode == 3: Single plot of "signal-noise".
    KLTmode : int
        Controls how noise affects output, for KLT.
        KLTmode == 0: Does not compute KLTs. (DEFAULT)
        KLTmode == 1: Plots signal only.
        KLTmode == 2: Separate plots for signal and noise.
        KLTmode == 3: Single plot of "signal-noise".
        KLTmode == 4: Single plot of signal and noise, for t=0.
        KLTmode == 5: Single plot of signal and noise, averaged.
        
    
    Returns:
    --------
    None

    '''

    frequencies = np.array([91.7,102.3,102.9,850.0,853.610,859.000,863.500,866.000])
    nsamples = frequencies.shape[0]

    if FFTmode!=0:
        for i in range(0,nsamples):
            spectrum,fmin,fmax,tmax = specplot(frequencies[i],noisemode=FFTmode,mode='FFT')
            print "Frequency = "+str(frequencies[i])+" MHz: FFT Complete"
    
    if KLTmode!=0:
        for i in range(0,nsamples):
            eigval,eigval_n,tmax = specplot(frequencies[i],noisemode=KLTmode,mode='KLT')
            print "Frequency = "+str(frequencies[i])+" MHz: KLT Complete"

def runsingle(frequency):
    '''
    Generates plots of KLTs and FFTs for a single
    frequency. Does, like, everything.
    
    Parameters:
    -----------
    frequency : float
        Frequency of sample (in MHz) that we want
        to run all available code for.
        
    
    Returns:
    --------
    None

    '''
    FFTmodes = 3    # (!) Maximum number for FFT's noisemode in specplot().
    KLTmodes = 5    # (!) Maximum number for KLT's noisemode in specplot().
    
    for i in range(2,FFTmodes+1):
        spectrum,fmin,fmax,tmax = specplot(frequency,noisemode=i,mode='FFT')
        print "Frequency = "+str(frequencies[i])+" MHz: Mode "+str(i)+" FFT Complete"
    
    for i in range(2,KLTmodes+1):
        eigval,eigval_n,tmax = specplot(frequency,noisemode=i,mode='KLT')
        print "Frequency = "+str(frequencies[i])+" MHz: Mode "+str(i)+" KLT Complete"
            
    

