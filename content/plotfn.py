import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class SimulatedFilter:
        
    def wrap_phase(self,phase):
        return (phase + np.pi) % (2 * np.pi) - np.pi

    def freqAnalysisPlot(self,title,graphsFilename,maincols,redpitayafilename,cols,seperator,seperator_redpiataya):        
        df = pd.read_csv(graphsFilename, sep=seperator, engine='python', usecols=maincols)
        df_redpitaya = pd.read_csv(redpitayafilename, sep=seperator_redpiataya, engine='python', usecols=cols)
        
        # Rename columns    
        df.columns = ['Frequency', 'Phase', 'Gain']    
        df_redpitaya.columns = ['Frequency', 'Gain', 'Phase']
        df = df.apply(pd.to_numeric, errors='coerce')
        df_redpitaya = df_redpitaya.apply(pd.to_numeric, errors='coerce')

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot Gain
        line1, = ax1.plot(df['Frequency'], df['Gain'], 'b-', label='Simulated  Gain /dB')    
        line3, = ax1.plot(df_redpitaya['Frequency'], df_redpitaya['Gain'], 'g-', label='Experimantal Gain /dB')    
        ax1.set_xlabel('Frequency /Hz')
        ax1.set_ylabel('Gain /dB')
        ax1.set_xscale('log')
        # ax1.tick_params(axis='y', labelcolor='b')

        # Plot Phase on secondary y-axis
        ax2 = ax1.twinx()
        line2, = ax2.plot(df['Frequency'],df['Phase'], 'r--', label='Simulated Phase /°')
        line4, = ax2.plot(df_redpitaya['Frequency'],df_redpitaya['Phase'], 'y--', label='Experimental Phase /°')
        ax2.set_ylabel('Phase /°')
        # ax2.tick_params(axis='y', labelcolor='r')

        # Combine legends from both axes
        lines = [line1, line2,line3,line4]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='lower right')

        # Title and layout
        plt.title(title)
        fig.tight_layout()
        plt.grid(True, linestyle='-', linewidth=0.5)
        plt.show()

    def transcientAnalysisPlot(self,title,idealGraphsFilename,cols):
        filename = idealGraphsFilename
        df = pd.read_csv(filename, sep=',', engine='python', usecols=cols)

        #BPF

        # Rename columns
        df.columns = ['time', 'input', 'output']
        df = df.apply(pd.to_numeric, errors='coerce')

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot Gain
        line1, = ax1.plot(df['time'], df['input'], 'g-', label='Volts (V)')    
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('Gain (dB)', color='b')    
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot Phase on secondary y-axis
        ax2 = ax1.twinx()
        line2, = ax2.plot(df['time'],df['output'], 'y-', label='Phase (°)')
        ax2.set_ylabel('Phase (°)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Combine legends from both axes
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper right')

        # Title and layout
        plt.title(title)
        fig.tight_layout()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    def generateBodePlots(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(my_path, "../content/RAW/idealGraphs.csv")
        lpf_filename = os.path.join(my_path, "../content/RAW/redpitayaGraphs/LPF.csv")
        hpf_filename = os.path.join(my_path, "../content/RAW/redpitayaGraphs/HPF.csv")
        bsf_filename = os.path.join(my_path, "../content/RAW/redpitayaGraphs/BSF.csv")
        bpf_filename = os.path.join(my_path, "../content/RAW/redpitayaGraphs/BPF.csv")        
        bpf_title = 'Frequency Response of BPF (Gain and Phase)'
        bsf_title = 'Frequency Response of BSF (Gain and Phase)'
        lpf_title = 'Frequency Response of LPF (Gain and Phase)'
        hpf_title = 'Frequency Response of HPF (Gain and Phase)'
        filters = {
            bpf_title : ([0,1,2],bpf_filename),
            bsf_title : ([0,3,4],bsf_filename),
            hpf_title : ([0,5,6],hpf_filename),
            lpf_title : ([0,7,8],lpf_filename)
        }
        for title_internal,attributes in filters.items():    
            self.freqAnalysisPlot(
                title=title_internal,
                maincols= attributes[0],
                cols=[0,1,2],
                graphsFilename=filename,
                redpitayafilename=attributes[1],
                seperator=';',
                seperator_redpiataya=','
            )
    def generateTranscientPlotsExperimental(self):
        my_path = os.path.abspath(os.path.dirname(__file__))        
        graphs = [
            (   
                "Band Pass",
                {
                    595 : os.path.join(my_path, "../content/RAW/Transcient_CSV/bpf_t_595.csv"),
                    1055: os.path.join(my_path, "../content/RAW/Transcient_CSV/bpf_t_1055.csv"),
                    2185: os.path.join(my_path, "../content/RAW/Transcient_CSV/bpf_t_2185.csv"), 
                    
                }
            ),
            (   
                "Band Stop",
                {
                    260: os.path.join(my_path, "../content/RAW/Transcient_CSV/bsf_t_260.csv"),
                    982: os.path.join(my_path, "../content/RAW/Transcient_CSV/bsf_t_982.csv"),
                    1055: os.path.join(my_path, "../content/RAW/Transcient_CSV/bsf_t_1055.csv"),
                    3187: os.path.join(my_path, "../content/RAW/Transcient_CSV/bsf_t_3187.csv"),
                }
            ),
            (   
                "High Pass",
                {
                    701: os.path.join(my_path, "../content/RAW/Transcient_CSV/hpf_t_701.csv"),
                    1043: os.path.join(my_path, "../content/RAW/Transcient_CSV/hpf_t_1043.csv"),
                    3232: os.path.join(my_path, "../content/RAW/Transcient_CSV/hpf_t_3232.csv"),
                }
            ),
            (   
                "Low Pass",
                {
                    390: os.path.join(my_path, "../content/RAW/Transcient_CSV/lpf_t_390.csv"),
                    1115: os.path.join(my_path, "../content/RAW/Transcient_CSV/lpf_t_1115.csv"),
                    1724: os.path.join(my_path, "../content/RAW/Transcient_CSV/lpf_t_1724.csv"),
                }
            )
        ]
        for title,graph in graphs:
            dfs=[]
            colors = ['b-', 'r-', 'g-', 'y-']
            for freq,filename in graph.items():
                dfs.append((pd.read_csv(filename, sep=",", engine='python', usecols=[0,1]),freq,colors.pop()))
            lines=[]        
            fig, ax1 = plt.subplots(figsize=(10, 6))    
            for df,freq,colors in dfs:
                # Rename columns
                df.columns = ['time', 'input']
                df = df.apply(pd.to_numeric, errors='coerce')
                # Plotting
                
                # Plot Gain
                line1, = ax1.plot(df['time'], df['input'], colors, label=f'{freq} /Hz')    
                ax1.set_xlabel('time /s')
                ax1.set_ylabel('Voltage /V', color='b')    
                ax1.tick_params(axis='y', labelcolor='b')
                # Combine legends from both axes
                lines.append(line1)
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='upper right')
            # Title and layout
            plt.title(f"Red Pitaya {title} Transcient Plots")
            fig.tight_layout()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.show()

    def generateTranscientPlotsSimulated(self):
        my_path = os.path.abspath(os.path.dirname(__file__))       
         
        # filename = os.path.join(my_path, "../content/RAW/kiCADTranscients/keycastranscients.csv"),
        filename = r"C:\Users\91854\Documents\Study Materials\summer25\amcd-shared-repo\hsb-amcd-sose2025-5\content\RAW\kiCADTranscients\keycastranscients.csv"
          
        df = pd.read_csv(filename, sep=";", engine='python', usecols=[0,1,2,3,4])            
        # Rename columns    
        df.columns = ['Time', 'Voltage(BPF)', 'Voltage(BSF)','Voltage(HPF)','Voltage(LPF)']                    
        df = df.apply(pd.to_numeric, errors='coerce')                

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot Gain
        line1, = ax1.plot(df['Time'], df['Voltage(BPF)'], 'b-', label='Voltage(BPF) /V')   
        line2, = ax1.plot(df['Time'], df['Voltage(BSF)'], 'g-', label='Voltage(BSF) /V')   
        line3, = ax1.plot(df['Time'], df['Voltage(HPF)'], 'r-', label='Voltage(HPF) /V')   
        line4, = ax1.plot(df['Time'], df['Voltage(LPF)'], 'y-', label='Voltage(LPF) /V')   
        ax1.set_xlabel('Time /s')
        ax1.set_ylabel('Voltage /V')
        # Combine legends from both axes
        lines = [line1,line2,line3,line4]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='lower right')

        # Title and layout
        plt.title(f"Step Analysis for Filters")
        fig.tight_layout()
        plt.grid(True, linestyle='-', linewidth=0.5)
        plt.show()       
        # for name,cols in filterdict:
        #     dfs=[]
        #     colors = ['b-', 'r-', 'g-', 'y-']
        #     for freq,filename in graph.items():
        #         dfs.append((pd.read_csv(filename, sep=";", engine='python', usecols=cols),freq,colors.pop()))
        #     lines=[]        
        #     fig, ax1 = plt.subplots(figsize=(10, 6))    
        #     for df,freq,colors in dfs:
        #         # Rename columns
        #         df.columns = ['time', 'input']
        #         df = df.apply(pd.to_numeric, errors='coerce')
        #         # Plotting
                
        #         # Plot Gain
        #         line1, = ax1.plot(df['time'], df['input'], colors, label=f'{freq} /Hz')    
        #         ax1.set_xlabel('time /s')
        #         ax1.set_ylabel('Voltage /V', color='b')    
        #         ax1.tick_params(axis='y', labelcolor='b')
        #         # Combine legends from both axes
        #         lines.append(line1)
        #     labels = [line.get_label() for line in lines]
        #     ax1.legend(lines, labels, loc='upper right')
        #     # Title and layout
        #     plt.title(f"Red Pitaya {name} Transcient Plots")
        #     fig.tight_layout()
        #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        #     plt.show()     



# SimulatedFilter().generateBodePlots()
# SimulatedFilter().generateTranscientPlotsExperimental()
SimulatedFilter().generateTranscientPlotsSimulated()

#------------------------------------------------------------------------------------------------------

# ideal Filter
import numpy as np
from scipy import signal
from scipy.signal import tf2zpk

class TheoreticalFilter:          
    def __init__(self,h_0,w_0,Q):                
        self.h_0 = h_0
        self.w_0 = w_0
        self.Q = Q                
    def lowpassfilter(self):        
        '''Low Pass Filter'''
        numerator = self.h_0
        numerator_string = f"          {numerator}"        
        return (numerator,*self.tf("Low Pass Filter: ",numerator_string))
    def highpassfilter(self):
        '''High Pass Filter'''        
        numerator = self.h_0*np.square(1/self.w_0)
        numerator_string = f"      {np.round(numerator*10**8,4)}e8 s^2"
        return (float(numerator),*self.tf("High Pass Filter: ",numerator_string))
    def bandpassfilter(self):        
        '''Band Pass Filter'''
        numerator = -(self.h_0)*1/self.w_0
        numerator_string = f"{np.round((numerator)*10**3,4)}e-3 s"
        return (numerator,*self.tf("Band Pass Filter: ",numerator_string))
    def bandstopfilter(self): 
        '''Band Stop Filter'''       
        numerator_part_a = self.h_0
        numerator_part_b = np.square(1/self.w_0)*self.h_0
        numerator_string = f"  ( {1} + {np.round(numerator_part_b*10**8,4)}e-8 s^2 ) {self.h_0}"        
        numerator = (float(numerator_part_a),float(numerator_part_b))
        return (*numerator,*self.tf("Band Stop Filter: ",numerator_string))
    def tf(self,name,num):
        real_den_2 = 1/self.w_0**2
        real_den_1 = 1/(self.w_0*self.Q)
        '''Just a Display Function of a TF'''
        line = "----------------------------"
        den = f"{real_den_2}s^2 + {real_den_1}s + 1"
        print(f"\n{name}\n")
        print(num)            
        print(line)
        print(den)
        return (float(real_den_2),float(real_den_1))

    def rootplotgen(self):
        filterdctionary = {
            "Low Pass":self.lowpassfilter(),
            "High Pass":self.highpassfilter(),
            "Band Pass":self.bandpassfilter(),
            "Band Stop":self.bandstopfilter(),
        }
        for name,func in filterdctionary.items():
            self.bodeplotgen(func,name)
            self.stepplotgen(func,name)
            self.pzplot(func,name)

    def bodeplotgen(self,values,filtername):        
        match filtername:
            case "Low Pass":                
                num_val,den2,den1 = values
                num = [num_val]
            case "High Pass":
                num_val,den2,den1 = values        
                num = [num_val,0,0]
            case "Band Pass":
                num_val,den2,den1 = values        
                num = [0,num_val,0]
            case "Band Stop":
                num_val_a,num_val_b,den2,den1 = values        
                num = [num_val_b,0,num_val_a]
        den = [den2,den1,1]
        print(num,den)
        sys = signal.TransferFunction(num, den)
        w, mag, phase = signal.bode(sys)
        
        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot Gain
        line1, = ax1.semilogx(w,mag, 'b-', label='Gain /dB')            
        ax1.set_xlabel('Frequency /Hz')
        ax1.set_ylabel('Gain /dB')        
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot Phase on secondary y-axis
        ax2 = ax1.twinx()
        line2, = ax2.semilogx(w,phase, 'r--', label='Phase /°')        
        ax2.set_ylabel('Phase /°')
        ax2.tick_params(axis='y', labelcolor='r')

        # Combine legends from both axes
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='lower left')

        # Title and layout
        plt.title(f"Theoretical  Phase and Magnitude plot of {filtername} Filter")
        fig.tight_layout()
        plt.grid(True, linestyle='-', linewidth=0.5)
        plt.show()  

    def stepplotgen(self,values,filtername):
        match filtername:
            case "Low Pass":                
                num_val,den2,den1 = values
                num = [num_val]
            case "High Pass":
                num_val,den2,den1 = values        
                num = [num_val,0,0]
            case "Band Pass":
                num_val,den2,den1 = values        
                num = [0,num_val,0]
            case "Band Stop":
                num_val_a,num_val_b,den2,den1 = values        
                num = [num_val_b,0,num_val_a]
        den = [den2,den1,1]

        sys = signal.TransferFunction(num, den)
        time, y = signal.step(sys)
        
        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot Gain
        line1, = ax1.plot(time, y , 'b-', label='Gain /dB')            
        ax1.set_xlabel('time /s')
        ax1.set_ylabel('Voltage /V')        
        # ax1.tick_params(axis='y', labelcolor='b')

        # Combine legends from both axes
        lines = [line1]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='lower left')

        # Title and layout
        plt.title(f"Theoretical Step Response plot of {filtername} Filter")
        fig.tight_layout()
        plt.grid(True, linestyle='-', linewidth=0.5)
        plt.show()

    def pzplot(self,values,filtername):
        match filtername:
            case "Low Pass":                
                num_val,den2,den1 = values
                numerator = [num_val]
            case "High Pass":
                num_val,den2,den1 = values        
                numerator = [num_val,0,0]
            case "Band Pass":
                num_val,den2,den1 = values        
                numerator = [0,num_val,0]
            case "Band Stop":
                num_val_a,num_val_b,den2,den1 = values        
                numerator = [num_val_b,0,num_val_a]
        denominator = [den2,den1,1]

        # Compute poles and zeros
        zeros, poles, gain = tf2zpk(numerator, denominator)

        # Plot the poles and zeros
        plt.figure(figsize=(10, 6))
        plt.scatter(np.real(zeros), np.imag(zeros), s=100, label='Zeros', marker='o', facecolors='none', edgecolors='b')
        plt.scatter(np.real(poles), np.imag(poles), s=100, label='Poles', marker='x', color='r')

        # Add grid and axis labels
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Pole-Zero Plot for {filtername} filter')
        plt.legend()
        plt.axis('equal')
        plt.show()        
# Given
Q = 10
w_0 =1000
# Choice
H_0 = 1 #gain
R = 1000 # Ohm
## so,
C = 1/(w_0*R)

### Rough Notes
BSF_resistor = Q*R
H_0_resistor = R/H_0
print(f"General Variables:\
R : {R*10**-3} K Ohm\n\
C : {np.round(C*10**6,3)} uF\n\
BSF resistor : {BSF_resistor*10**-3} kOhm,\n\
General R: {R*10e-3}kOhm\n\
H_0 Resistor: {H_0_resistor*10**-3}kOhm"
)

# # Filter design
# idealfilter = TheoreticalFilter(H_0,w_0,Q)
# print("Transfer Functions:")
# idealfilter.rootplotgen()
