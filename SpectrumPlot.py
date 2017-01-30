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
    '''

    if preset==True:
	frequency,Nchannel,rate = loadcustom(frequency)          # Radio freq. in MHz; No. of samples; sampling freq. in MHz.
	outputsuffix = str(frequency).replace(".","_")     # e.g. if freq = 96.5, then outputsuffix = '96_5'.

    # Reads data file:
    spec, freq, fftd, time = readradio(frequency = frequency*u.MHz, Nchannel=Nchannel, \
                                  rate=rate*u.MHz, fname = "radioline_"+outputsuffix+".bin")
	

    # PLOTTING: Spectrum (intensity vs frequency) vs Time

    dt = (Nchannel/(rate*1e6)) # Size of time step, in seconds.

    xmin=freq[0].value
    xmax=freq[-1].value
    ymin=0
    ymax= dt * fftd.shape[0]

    plt.imshow(np.log(np.abs(fftd)), extent = [xmin,xmax,ymin,ymax], aspect='auto', origin='lower')
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (s)")
    plt.title("Spectrum vs Time")

    plt.savefig('spec_vs_time_'+outputsuffix+'_MHz.png')
