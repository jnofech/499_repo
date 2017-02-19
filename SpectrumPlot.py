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
        Returns the raw array if True. False by
        default.
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
            2D array of _complex_ power(?) vs frequency,
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
        time : array
            1D array of corresponding time.
    '''
    d = np.fromfile(fname,dtype=np.uint8)
    d = d.astype(np.float)
    d -= 127
    d = d[0::2]+d[1::2]*1j
    if timesamples:
        return(d)
    d.shape = (d.size/Nchannel,Nchannel) # Navgs by Nchannels
    

    if mode=='fft' or mode=='FFT':
        # Fast Fourier Transform
        fftd = np.fft.fft(d,axis=1)           # (?) Spectrum?
        spec = np.mean(np.abs(fftd),axis=0)   # (?) Time-averaged spectrum vs frequency?
        
        freq = np.fft.fftfreq(Nchannel)*rate+frequency
        print d.size
        time = (np.arange(d.size/Nchannel)/rate)

        freq = np.roll(freq, np.size(freq)/2)         # Array of frequencies, fixed to range from lowest to highest.
        spec = np.roll(spec, np.size(spec)/2)         # Array of corresponding average spectrum, fixed the same way.
        fftd = np.roll(fftd, np.size(freq)/2, axis=1) # Array of corresponding spectrum AT DIFFERENT TIMES, also fixed.
        
        return spec,freq,fftd,time
        
    elif mode=='klt' or mode=='KLT':
        folds = 10       # (!) Change this manually if you want to run it faster or slower. Recommended: 5.
        step = 10         # (!) Change this manually if you want to run it faster, at the cost of imshow resolution.
                         #        Default: 1.
        
        # Let's reshape the signal into  
        d.shape = (d.shape[0]*folds, d.shape[1]/folds)
        # KL Transform
        ffty = np.fft.fft(d, axis=0)
        acorfft = np.fft.ifft(ffty * np.conj(ffty), axis=0)
        # Drop the imaginary component.
        acorfft = acorfft.real
        # Magic of the KL Transform is just to calculate the eigenvalues of the Toeplitz matrix.
        print acorfft.size
        print acorfft.shape
        eigval = np.zeros(acorfft.size/step).reshape(acorfft.shape[0]/step,acorfft.shape[1]) # Creates empty array of same size.
        
        for i in range(0,acorfft.shape[0]/step):
            toeplitz_matrix = sla.toeplitz(acorfft[i])
            print i
            eigval[i], dummy = np.linalg.eigh(toeplitz_matrix)  # Don't bother with eigenvectors.

        time = (np.arange(d.size/Nchannel)/rate)
        
        return eigval,time
    
    else:
        print "ERROR : 'mode' must equal FFT' or 'KLT'."
        return


def savecustom(frequency, Nchannel, rate):
    # When saving, all values must be UNITLESS. Units will be applied in other functions.
    '''
	When saving, all values must be UNITLESS. Presets are saved according
		to frequency alone; i.e. you can not have multiple presets
		for the same frequency.


	Parameters:
	-----------
	frequency : float
		Frequency of sampled radio line, in MHz.
	Nchannel : int
		Number of samples.	
	rate : float
		Sampling rate, in MHz (millions of
		samples per second).
    '''
    
    custom_index=['frequency','Nchannel','rate']     	# Frequency of radio signal, number of samples, sampling rate.
    custom = [frequency, int(Nchannel), rate]
    
    outputsuffix = str(frequency).replace(".","_")     	# e.g. if freq = 96.5, then outputsuffix = '96_5'.
    outfile = "radioline_"+outputsuffix+".bin"

    f = file(str(outfile)+".bin","wb")			# Saves the following into 'filename.bin'.
    np.save(f,custom)					# Saves array1.
    np.save(f,custom_index)				# Saves array2.
    f.close()						# No more saving; back to regular code.
    
def loadcustom(frequency):
    # All output will be UNITLESS. Units will be applied in other functions.
    '''
	When loading, all values will be UNITLESS.


	Parameters:
	-----------
	frequency : float
    '''


    outputsuffix = str(frequency).replace(".","_")     # e.g. if freq = 96.5, then outputsuffix = '96_5'.
    outfile = "radioline_"+outputsuffix+".bin"

    f = file(str(outfile)+".bin","rb")			# Loads 'filename.bin'.
    custom = np.load(f)					# Loads the old 'array1' into ARRAY1.
    custom_index = np.load(f)				# Loads the old 'array2' into ARRAY2. Note that this is 
                                           		#     based on the order in which the arrays were saved! 
                                           		#     You don't get to specify which array to load
                                            		#     based on its name.
    f.close()						# No more loading; back to regular code.

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
    Nchannel : int
        Number of samples.
    rate : float
        Sampling rate, in MHz (millions of
        samples per second).
    preset : bool
        Toggles whether a preset is activated.
        Default: True
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
        tmax : float
            Time elapsed for sampling the data.
    '''

    if preset==True:
        frequency,Nchannel,rate = loadcustom(frequency)          # Radio freq. in MHz; No. of samples; sampling freq. in MHz.
    outputsuffix = str(frequency).replace(".","_")     # e.g. if freq = 96.5, then outputsuffix = '96_5'.


    if mode=='fft' or mode=='FFT':
        
        # Reads data file:
        spec, freq, fftd, time = readradio(frequency = frequency*u.MHz, Nchannel=Nchannel, \
                                      rate=rate*u.MHz, fname = "radioline_"+outputsuffix+".bin",mode='FFT')
        
        # PLOTTING: Spectrum (intensity vs frequency) vs Time

        dt = (Nchannel/(rate*1e6)) # Size of time step, in seconds.

        power = np.log(np.abs(fftd))
        fmin = freq[0].value
        fmax = freq[-1].value
        tmin = 0
        tmax = dt * power.shape[0]


        xmin=fmin
        xmax=fmax
        ymin=0
        ymax= tmax

        plt.imshow(power, extent = [xmin,xmax,ymin,ymax], aspect='auto', origin='lower')
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (s)")
        plt.title("Spectrum vs Time")

        plt.savefig('spec_vs_time_'+outputsuffix+'_MHz.png')

        return frequency,power,fftd,fmin,fmax,tmax
    elif mode=='klt' or mode=='KLT':
        
        # Reads data file:
        eigval, time = readradio(frequency = frequency*u.MHz, Nchannel=Nchannel, \
                                      rate=rate*u.MHz, fname = "radioline_"+outputsuffix+".bin",mode='KLT')
        
        # PLOTTING: Eigenvalues (eigval vs eigval number) vs Time

        dt = (Nchannel/(rate*1e6)) # Size of time step, in seconds.

        eigval = eigval[:,::-1]     #(?) Reverses the order of the eigenvalues, so that they descend rather than ascend.
        
        tmin = 0
        tmax = dt * eigval.shape[0]

        ymin=0
        ymax= tmax

    
        print eigval
        print eigval.shape
        vmax = np.average(np.abs(eigval))*10     # Maximum value that will be registered on the color bar.
        
        plt.imshow(eigval, extent = [0,eigval.shape[1],ymin,ymax], vmin=-1*vmax, vmax=vmax, aspect='auto', origin='lower')
        plt.xlabel("Eigenvalue Number")
        plt.ylabel("Time (s)")
        plt.title("Eigenvalues vs Time")
        plt.colorbar()

        plt.savefig('eig_vs_time_'+outputsuffix+'_MHz.png')
        
        return eigval, tmax
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
    power_n : array
        2D array of spectrum (log(abs(power))
        vs frequency) as a function of time,
        WITH noise applied.
    fftd_n : array
        2D array of _complex_ power vs frequency,
        as a function of time, WITH noise applied.
    noise : array
        2D array of _complex_ noise, added to fftd,
        should you feel like using it for some reason.
    '''

    noise_x = np.random.randn(fftd.shape[0],fftd.shape[1])    
    noise_y = np.random.randn(fftd.shape[0],fftd.shape[1])

    noise = scale*(noise_x + noise_y*1j)    # At the moment, the noise can be positive or negative;
                                            # as opposed to ranging from 0 to x. Is this okay?
    
    fftd_n = fftd + noise
    
    power_n = np.log(np.abs(fftd_n))
    
    return power_n, fftd_n, noise
