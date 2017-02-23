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
        spec : array
            1D array of time-averaged spectrum
            vs frequency.
        freq : array
            1D array of corresponding frequencies to
            spec and fftd.
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
        eigval_n : array
            2D array of eigenvalues vs eigenvalue
            number, as a function of time, for
            NOISE (as opposed to measured signal).
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
        spec = np.mean(np.abs(fftd),axis=0)   # Time-averaged spectrum vs frequency
        
        
        freq = np.fft.fftfreq(Nchannel)*rate+frequency
        print d.size
        time = (np.arange(d.size/Nchannel)/rate)

        freq = np.roll(freq, np.size(freq)/2)         # Array of frequencies, fixed to range from lowest to highest.
        spec = np.roll(spec, np.size(spec)/2)         # Array of corresponding average spectrum, fixed the same way.
        fftd = np.roll(fftd, np.size(freq)/2, axis=1) # Array of corresponding spectrum AT DIFFERENT TIMES, also fixed.
        
        return spec,freq,fftd,time
        
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
        eigval_n = addnoise_KLT(d,eigval,step,scale)
        
        return eigval,eigval_n,time
        #return eigval,time
    
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

def specplot(frequency, Nchannel=5000, rate=3, preset=True, mode='FFT'):
    '''
    Plots spectrum or eigenvalues versus time from a given data file, then
    saves it.

    Parameters:
    -----------
    frequency : float
        Frequency of sampled radio line, in MHz.
        A frequency of 0 MHz will be interpreted as
        blank noise.
    Nchannel : int
        Number of samples.
    rate : float
        Sampling rate, in MHz (millions of
        samples per second).
    preset : bool
        Toggles whether a preset is activated.
        Default: True
        (!) False is currently unsupported.
    mode : str ('FFT' or 'KLT')
        Analyzes data using fast fourier transform or
        Karhunen-Loeve transform.
        
    Returns:
    --------
    if mode=='FFT':
        frequency : float
            Same as input. It's returned for convenience
            if you want to follow up with the 'disperser'
            function.
        power : array
            2D array of spectrum (log(abs(power))
            vs frequency) as a function of time.
        fftd : array
            2D array of _complex_ power vs frequency,
            as a function of time.
        f_min : float
            Minimum and maximum values for frequency
            in the 'power' array.
        fmax : float
            Minimum and maximum values for frequency
            in the 'power' array.
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

    if preset==True:
        frequency,Nchannel,rate = loadcustom(frequency)          # Radio freq. in MHz; No. of samples; sampling freq. in MHz.
    outputsuffix = str(frequency).replace(".","_")     # e.g. if freq = 96.5, then outputsuffix = '96_5'.


    if mode=='fft' or mode=='FFT':
        
        # Reads data file:
        spec, freq, fftd, time = readradio(frequency = frequency*u.MHz, Nchannel=Nchannel, \
                                      rate=rate*u.MHz, fname = "radioline_"+outputsuffix+".bin",\
                                      mode='FFT')
        # Generates noise array:
        fftd_n, noise, power_n = addnoise(fftd,1)
        
        
        # PLOTTING: Spectrum (intensity vs frequency) vs Time

        dt = (Nchannel/(rate*1e6)) # Size of time step, in seconds.

        power = np.log(np.abs(fftd))
        noisepower = np.log(np.abs(noise))
        fmin = freq[0].value
        fmax = freq[-1].value
        tmin = 0
        tmax = dt * power.shape[0]


        xmin=fmin
        xmax=fmax
        ymin=0
        ymax= tmax

        # Draw Spectrum vs. Time for SIGNAL
        plt.imshow(power, extent = [xmin,xmax,ymin,ymax], aspect='auto', origin='lower')
        plt.colorbar()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (s)")
        plt.title("Spectrum vs Time, for "+str(frequency)+" MHz")

        plt.savefig('spec_vs_time_'+outputsuffix+'_MHz.png')
        plt.clf()
        
        # Draw Spectrum vs. Time for NOISE
        plt.imshow(noisepower, extent = [xmin,xmax,ymin,ymax], aspect='auto', origin='lower')
        plt.colorbar()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (s)")
        plt.title("Spectrum vs Time ("+str(frequency)+" MHz), for NOISE")

        plt.savefig('spec_vs_time_'+outputsuffix+'_MHz_n.png')
        plt.clf()
        return frequency,power,fftd,fmin,fmax,tmax
    elif mode=='klt' or mode=='KLT':
        # (!) Change these parameters.
        Nmax = 100       # Number of eigenvalues to be displayed on x-axis.
        
        # Reads data file:
        eigval, eigval_n, time = readradio(frequency = frequency*u.MHz, Nchannel=Nchannel, \
                                      rate=rate*u.MHz, fname = "radioline_"+outputsuffix+".bin",\
                                      mode='KLT')
        
        # PLOTTING: Eigenvalues (eigval vs eigval number) vs Time

        dt = (Nchannel/(rate*1e6)) # Size of time step, in seconds.
        
        tmin = 0
        tmax = dt * eigval.shape[0]
        ymin=0
        ymax= tmax

        eigval = np.abs(eigval)     # Takes absolute value of eigenvalues.
        eigval = np.sort(eigval)    # Lists |eigval| in ascending order.
        eigval = eigval[:,::-1]     # Reverses the order of |eigval|, so that they descend rather than ascend.
        
        eigval_n = np.abs(eigval_n) # Repeat for eigval_n.
        eigval_n = np.sort(eigval_n)#
        eigval_n = eigval_n[:,::-1] #  
        
        vmax = np.average(eigval)*10     # Maximum value that will be registered on the color bar.
        vmax_n = np.average(eigval_n)*10 # Same, but for noise.
        
        
        # Plotting & saving, for signal.
        plt.imshow(eigval[:,:Nmax], extent = [0,Nmax,ymin,ymax], \
                   vmin=-1*vmax, vmax=vmax, aspect='auto', origin='lower')
        plt.xlabel("Eigenvalue Number")
        plt.ylabel("Time (s)")
        plt.title("Eigenvalues vs Time, for "+str(frequency)+" MHz")
        plt.colorbar()

        plt.savefig('eig_vs_time_'+outputsuffix+'_MHz.png')
        plt.clf()
        
        
        # Plotting & saving, for noise.
        plt.imshow(eigval_n[:,:Nmax], extent = [0,Nmax,ymin,ymax], \
                   vmin=-1*vmax_n, vmax=vmax_n, aspect='auto', origin='lower')
        plt.xlabel("Eigenvalue Number")
        plt.ylabel("Time (s)")
        plt.title("Eigenvalues vs Time ("+str(frequency)+" MHz), for NOISE ONLY.")
        plt.colorbar()

        plt.savefig('eig_vs_time_'+outputsuffix+'_MHz_n.png')
        plt.clf()
        
        return eigval, eigval_n, tmax
    else:
        print "ERROR : 'mode' must equal FFT' or 'KLT'."
        return
    
    

def disperser(frequency,power,fmin,fmax,tmax,DM):
    '''
    Shifts the elapsed time of a plot of spectrum
    vs time, as if it had passed through a plasma of 
    dispersion measure 'DM'.
    
    Parameters:
    -----------
    frequency : float
        Frequency of the sampled radio line,
        in MHz.
    power : array
        2D array of spectrum (log(abs(power))
        vs frequency) as a function of time.
    f_min : float
        Minimum and maximum values for frequency
        in the 'power' array.
    fmax : float
        Minimum and maximum values for frequency
        in the 'power' array.
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
    
    frange = np.linspace(fmin,fmax,power.shape[1])    # Array containing the range of frequencies.
    tstep = tmax / power.shape[0]                     # Time per "step" in array, in seconds
    
    tau = q_e**2 / (2 * np.pi * c_l * m_e * (frange*1e6)**2) * DM * 3.086e16
                                                                     # Time delayed by the dispersion measure DM
                                                                     #(?) Does the first L/c_l term get ignored?
    # tau = np.zeros(power.shape[1]) + np.random.rand(power.shape[1])*0.5   # Testing the code for larger 'tau'.
    
    
    # Expanding Array by As Large as Necessary
    
    n_rows = np.int(np.max(tau) / tstep)        # Number of extra rows the array needs
    blankspace = np.zeros(power.shape[1]*n_rows).reshape(n_rows,power.shape[1])
    power = np.vstack([power,blankspace])
    
    # SHIFTING the Array by Tau
    tau_step = tau / tstep              # tau, in units of steps
    tau_step = tau_step.astype(int)
    
    for i in range(0,power.shape[1]):
        power[:,i] = np.roll(power[:,i],tau_step[i])
    
    
    # PLOTTING: Spectrum (intensity vs frequency) vs Time    
    
    xmin=fmin
    xmax=fmax
    ymin=0
    ymax= tmax + np.max(tau)

    plt.imshow(power, extent = [xmin,xmax,ymin,ymax], aspect='auto', origin='lower')
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (s)")
    plt.title("Spectrum vs Time")

    plt.savefig('spec_vs_time_'+outputsuffix+'_MHz_DM_is_'+str(DM)+'.png')
    
    return tau


def addnoise(fftd,scale=1):
    '''
    Takes array of _complex_ power vs
    frequency, then adds random noise with a
    standard deviation of 1*scale to it.

    Parameters:
    -----------
    fftd : array
        2D array of _complex_ power vs frequency,
        as a function of time.
    scale : float
        Multiplier for standard deviation of noise.
        e.g. scale==2 will cause noise to have a 
        standard deviation of 2.
        
    Returns:
    --------
    fftd_n : array
        2D array of _complex_ power vs frequency,
        as a function of time, WITH noise applied.
    noise : array
        2D array of _complex_ noise, added to fftd,
        should you feel like using it for some reason.
    power_n : array
        2D array of spectrum (log(abs(power))
        vs frequency) as a function of time,
        WITH noise applied.
    '''

    noise_x = np.random.randn(fftd.shape[0],fftd.shape[1])    # Real components of noise.
    noise_y = np.random.randn(fftd.shape[0],fftd.shape[1])    # Imaginary components of noise.

    noise = scale*(noise_x + noise_y*1j)    # At the moment, the noise can be positive or negative;
                                            # as opposed to ranging from 0 to x. Is this okay?
#    # Multiply the noise by a factor.
#    Pnoise = noise.shape[1]           # (?) "Pnoise = Npts * 1", according to instructions.
#                                      #     Is Npts simply the number of entries in a "power vs frequency" row?
#    
#    spec = np.ave(np.abs(fftd),axis=0)      # Time-averaged spectrum (Power vs. Frequency).
#    Psig = np.sum(np.abs(spec)**2)
#    print Pnoise, Psig
#    noise = noise * np.sqrt(Psig/Pnoise)
    
    # Generate the fftd+noise array.
    fftd_n = fftd + noise
    
    power_n = np.log(np.abs(fftd_n))
    
    return fftd_n, noise, power_n

def addnoise_KLT(d,eigval,step=1,scale=1):
    '''
    Takes array of (eigval. vs frequency)
    vs time, then generates a similar array
    using pure noise (with a standard 
    deviation of 1*scale to it).

    Parameters:
    -----------
    d : array
        Signal array, after being reshaped
        from folding.
    eigval : array
        2D array of eigenvalues vs eigenvalue
        number, as a function of time.
    step : int
        Step size for generating eigval array.
        A step size of 1 (default) will often be
        far too slow, and can cause imshow to freeze.
    scale : float
        Multiplier for standard deviation of noise.
        e.g. scale==2 will cause noise to have a 
        standard deviation of 2.
        
    Returns:
    --------
    noise : array
        2D array of _complex_ signal vs. time, from
        noise alone. 
    eigval_n : array
        2D array of eigenvalues vs eigenvalue number
        _from noise only_, as a function of time.
    '''
    noise_x = np.random.randn(d.shape[0],d.shape[1])    # Real components of noise.
    noise_y = np.random.randn(d.shape[0],d.shape[1])    # Imaginary components of noise.

    noise = scale*(noise_x + noise_y*1j)    # Noise can be + or -.
    
                # KLT THE DATA
    Pnoise = noise.shape[1] 
    Psig = np.zeros(d.shape[0])   # Will be filled.

    acorfft_n = np.fft.ifft(noise * np.conj(noise), axis=0)   # Will be overwritten in a few lines.
    
    print acorfft_n.shape
    print d.shape
    print Psig.shape
    # Altering the noise array:
    for i in range(0,acorfft_n.shape[0]):
        Psig[i] = np.sum(np.abs(d[i])**2)
        noise[i] = noise[i] * np.sqrt(Psig[i]/Pnoise)
    
    # Generating eigval_n:
    acorfft_n = np.fft.ifft(noise * np.conj(noise), axis=0)
    acorfft_n = acorfft_n.real
    eigval_n = np.zeros(acorfft_n.size/step).reshape(acorfft_n.shape[0]/step,acorfft_n.shape[1])
    for i in range(0,acorfft_n.shape[0],step):
        toeplitz_matrix_n = sla.toeplitz(acorfft_n[i])
        eigval_n[i/step], dummy = np.linalg.eigh(toeplitz_matrix_n)  # Don't bother with eigenvectors.
    
    
    return eigval_n


