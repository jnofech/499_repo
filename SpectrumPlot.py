import numpy as np
import astropy.units as u
import subprocess
from tempfile import TemporaryFile		# Enables file-saving/loading.
import matplotlib.pyplot as plt

# 1.29.17 - Manipulating and displaying USB radio antenna data. 

print('"SpectrumPlot.py" \n \nAvailable functions: \n  readradio - Reads the data from "sampler"\'s output.\n  savecustom - Saves a custom preset for frequency, sample frequency, etc.\n  loadcustom - Loads a custom preset.\n  specplot - Plots the spectrum vs time of a given frequency\'s data file.')


def readradio(fname='radioline.bin',Nchannel=5000,frequency = 96.7*u.MHz, rate = 3*u.MHz, timesamples=False):
    '''
	Reads from a saved data file.

	
	Parameters:
	-----------
	fname : str
		Name of data file. Must include '.bin'
		suffix.
	Nchannel : int
		Number of samples.
	frequency : float
		Frequency of sampled radio line, in MHz.
	rate : float
		Sampling rate, in MHz (millions of
		samples per second).
	timesamples : bool
		Toggles whether to return raw data.
		FALSE (disabled) by default.

	
	Returns:
	--------
	spec : array
		1D array of average relative intensity.
	freq : array
		1D array of corresponding frequency in MHz.
	fftd : array
		2D array of corresponding spectrum versus time.	
	time : array
		??????
    '''
    d = np.fromfile(fname,dtype=np.uint8)
    d = d.astype(np.float)
    d -= 127
    d = d[0::2]+d[1::2]*1j
    if timesamples:
        return(d)
    d.shape = (d.size/Nchannel,Nchannel) # Navgs by Nchannels
    fftd = np.fft.fft(d,axis=1)           # (?) Spectrum?
    spec = np.mean(np.abs(fftd),axis=0)   # (?) Time-averaged spectrum vs frequency?
    freq = np.fft.fftfreq(Nchannel)*rate+frequency
    print freq.shape
    time = (np.arange(d.size/Nchannel)/rate)
    
    freq = np.roll(freq, np.size(freq)/2)         # Array of frequencies, fixed to range from lowest to highest.
    spec = np.roll(spec, np.size(spec)/2)         # Array of corresponding average spectrum, fixed the same way.
    fftd = np.roll(fftd, np.size(freq)/2, axis=1) # Array of corresponding spectrum AT DIFFERENT TIMES, also fixed.
    return(spec,freq,fftd,time)
    print(fftd.shape)    

    return(spec,freq,fftd,time)



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

def specplot(frequency, Nchannel=5000, rate=3, preset=True):
    '''
	Plots spectrum versus time from a given data file, then
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
	custom : bool
		Toggles whether a preset is activated.
		Default: True
        
    Returns:
    --------
    frequency : float
        Same as input. It's returned for convenience
        if you want to follow up with the 'disperser'
        function.
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
        Time elapsed for sampling the data.
    '''

    if preset==True:
        frequency,Nchannel,rate = loadcustom(frequency)          # Radio freq. in MHz; No. of samples; sampling freq. in MHz.
    outputsuffix = str(frequency).replace(".","_")     # e.g. if freq = 96.5, then outputsuffix = '96_5'.

    # Reads data file:
    spec, freq, fftd, time = readradio(frequency = frequency*u.MHz, Nchannel=Nchannel, \
                                  rate=rate*u.MHz, fname = "radioline_"+outputsuffix+".bin")


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
    
    return frequency,power,fmin,fmax,tmax


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
    tau : float
	The time delay from passing the signal
	through a cloud of dispersion measure DM.
    '''
    
    outputsuffix = str(frequency).replace(".","_")     # e.g. if freq = 96.5, then outputsuffix = '96_5'.

    q_e = 1.602e-19     # Elementary charge, in C
    c_l = 3.0e10        # Speed of light, in cm/s
    m_e = 9.109e-28     # Mass of electron, in g
    
    tau = q_e**2 / (2 * np.pi * c_l * m_e * (frequency*1e6)**2) * DM    # Time delayed by the dispersion measure DM
                                                                  #(?) Does the first L/c_l term get ignored?
    
    
    # PLOTTING: Spectrum (intensity vs frequency) vs Time    
    
    xmin=fmin
    xmax=fmax
    ymin=0
    ymax= tmax + tau

    plt.imshow(power, extent = [xmin,xmax,ymin,ymax], aspect='auto', origin='lower')
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (s)")
    plt.title("Spectrum vs Time")
    
    print tau

    plt.savefig('spec_vs_time_'+outputsuffix+'_MHz_DM_is_'+str(DM)+'.png')
    
    return tau

