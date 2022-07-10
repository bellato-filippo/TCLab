import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

#-----------------------------------
def myRound(x, base=5):
    
    """
    Returns a float that is the closest multiple of "base" near "x"
    Based on: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    
    :x: parameter that is rounded to a multiple of "base"
    :base: "base" parameter (optional: default value is 5)
    
    :return: rounded parameter
    """
    
    return float(base * round(float(x)/base))

#-----------------------------------
def SelectPath_RT(path,time,signal):
    
    """
    The function "SelectPath_RT" needs to be included in a "for or while loop".
    
    :path: dictionary input describing a path in time. Example: path = {0: 0, 5: 1, 50: 2, 80: 3, 100: 3}
    :time: time vector.
    :signal: signal vector that is being constructed using the input "path" and the vector "time".
    
    The function "SelectPath_RT" takes the last element in the vector "time" and, given the input "path", it appends the correct value to the vector "signal".
    """    
    
    for timeKey in path:
        if(time[-1] >= timeKey):
            timeKeyPrevious = timeKey    
    
    value = path[timeKeyPrevious]
    signal.append(value)

#-----------------------------------
def Delay_RT(MV,theta,Ts,MV_Delay,MVInit=0):
    
    """
    The function "Delay_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :theta: delay [s]
    :Ts: sampling period [s]
    :MV_Delay: delayed input vector
    :MVInit: (optional: default value is 0)
    
    The function "Delay_RT" appends a value to the vector "MV_Delay".
    The appended value corresponds to the value in the vector "MV" "theta" seconds ago.
    If "theta" is not a multiple of "Ts", "theta" is replaced by Ts*int(np.ceil(theta/Ts)), i.e. the closest multiple of "Ts" larger than "theta".
    If the value of the vector "input" "theta" seconds ago is not defined, the value "MVInit" is used.
    """
    
    NDelay = int(np.ceil(theta/Ts))
    if NDelay > len(MV)-1:
        MV_Delay.append(MVInit)
    else:    
        MV_Delay.append(MV[-NDelay-1])

#-----------------------------------        
def LEAD_LAG_RT(MV, Kp, Tlead, Tlag, Ts, PV, PVInit=0, method='EBD'):
    
    """
    The function "LeadLag_RT" needs to be included in a "for or while loop". 

    :MV: input vector
    :Kp: process gain
    :Tlead: lead time constant [s]
    :Tlag: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoidal method
        
    The function appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (Tlag != 0):
        K = Ts/Tlag
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
                PV.append(((1 / (1 + K)) * PV[-1]) + ((Kp * K) / (1 + K)) * (((1 + (Tlead / Ts)) * MV[-1]) - ((Tlead / Ts) * MV[-2])))
            elif method == 'EFD':
                PV.append(((1 - K) * PV[-1]) + (K * Kp)*(((Tlead / Ts) * MV[-1]) + (1 - (Tlead / Ts)) * MV[-2]))
            elif method == 'TRAP':
                PV.append((1/(2*Tlag+Ts))*((2*Tlag-Ts)*PV[-1] + Kp*Ts*(MV[-1] + MV[-2])))            
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*MV[-1])
    else:
        PV.append(Kp*MV[-1])
        
#-----------------------------------
def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, method='EBD-EBD'):
    
    """
    The function "PID_RT" needs to be included in a "for or while loop".
    
    :SP: SP (or SetPoint) vector
    :PV: PV (or Process Value) vector
    :Man: Man (or Manual controller mode) vector (True or False)
    :MVMan: MVMan (or Manual value for MV) vector
    :MVFF: MVFF (or Feedforward) vector
    
    
    :Kc: controller gain
    :Ti: integral time constant [s]
    :Td: derivative time constant [s]
    :alpha: Tfd = alpha*Td where Tfd is the derivative filter time constant [s]
    :Ts: sampling period [s]
    
    :MVMin: minimum value for MV (used for saturation and anti wind-up)
    :MVMax: maximum value for MV (used for saturation and anti wind-up)
    
    :MV: MV (or Manipulated Value) vector
    :MVP: MVP (or Propotional part of MV) vector
    :MVI: MVI (or Integral part of MV) vector
    :MVD: MVD (or Derivative part of MV) vector
    :E: E (or control Error) vector
    
    :ManFF: Activated FF in manual mode (optional: default boolean value is False)
    :PVInit: Initial value for PV (optional: default value is 0): used if PID_RT is ran first in the squence and no value of PV is available yet.
    
    :method: discretisation method (optional: default value is 'EBD')
        EBD-EBD: EBD for integral action and EBD for derivative action
        EBD-TRAP: EBD for integral action and TRAP for derivative action
        TRAP-EBD: TRAP for integral action and EBD for derivative action
        TRAP-TRAP: TRAP for integral action and TRAP for derivative action
        
    The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI", and "MVD".
    The appended values are based on the PID algorithm, the controller mode, and feedforward.
    Note that saturation of "MV" within the limits [MVMin MVMax] is implemented with anti wind-up. 
    """
    #create Tfd variable as alpha * Tderivative
    #Ts / 2 < Tfd
    Tfd = alpha * Td
    
    #split method string into integral method and derivative method
    methodI, methodD = method.split('-')
    
    #initialization of E
    if (len(PV) == 0):
        E.append(SP[-1] - PVInit)
    else:
        E.append(SP[-1] - PV[-1])
        
    #initialization of MVP
    MVP.append(Kc * E[-1])
    
    #initialization of MVD
    if (len(MVD) == 0):
        MVD.append((Kc * Td) / (Tfd + Ts) * (E[-1]))
    else:
        if (methodD == 'TRAP'):
            MVD.append((((Tfd - Ts * 0.5) / (Tfd + Ts * 0.5)) * MVD[-1]) + ((Kc * Td / (Tfd + Ts * 0.5)) * (E[-1] - E[-2])))
        else:
            MVD.append((((Tfd) / (Tfd + Ts)) * MVD[-1]) + (((Kc * Td) / (Tfd + Ts)) * (E[-1] - E[-2])))
    
    #initialization of MVI
    if (len(MVI) == 0):
        MVI.append((Kc * Ts / Ti) * E[-1])
    else:
        if (methodI == 'TRAP'):
            MVI.append(MVI[-1] + (0.5 * Kc * Ts / Ti) * (E[-1] + E[-2]))
        else:
            MVI.append(MVI[-1] + (Kc * Ts / Ti) * E[-1])
    
    # calculate MV, saturation and anti wind-up
    if (Man[-1] == True):
        MVI[-1] = MVI[-2]
        if (ManFF):
            if (MVMan[-1] + MVFF[-1] > MVMax):
                MV.append(MVMax)
            else:
                MV.append(MVMan[-1] + MVFF[-1])
        else:
            MV.append(MVMan[-1])
    elif (MVP[-1] + MVI[-1] + MVD[-1] > MVMax):
        MV.append(MVMax)
    elif (MVP[-1] + MVI[-1] + MVD[-1] < MVMin):
        MV.append(MVMin)
    elif (MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1] > MVMax):
        MV.append(MVMax)
    elif (MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1]< MVMin):
        MV.append(MVMin)
    else:
        if (ManFF):
            MV.append(MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1])
        else:
            MV.append(MVP[-1] + MVI[-1] + MVD[-1])
    
    
    
    
    
#--------------------
def IMC_TUNING(gamma, Kp, T1, T2, theta, method='FOPD-PID'):
    
    """
    
    :Kc: controller gain
    :T1: first time constant [s]
    :T2: second time constant [s]
    :T_olp: open-loop time constant [s]
    :Tc: closed-loop time constant [s]
    :gamma: Tc factor
    :Ti: integral time constant [s]
    :Td: derivative time constant [s]
    :Kp: process gain
    :theta: delay
    :method: type of the transfer function and control method (optional: default value is 'FOPD_PID')
        'FO-PI': first order, PI controller
        'SO-PID': second order, PID controller
        'FOPD-PI': first order plus delay, PI controller
        'FOPD-PID': first order plus delay, PID controller
        'SOPD-PID': second order plus delay, PID controller
        
    The function "IMC_TUNING" calculates the best PID parameters for the process
        
    """
    T_olp = max(T1, T2)
    Tc = gamma * T_olp
    Kc = 0
    Ti = 0
    Td = 0
    
    order, control = method.split('-')
    if (control == 'PI'):
        if (order == 'FO'):
            Kc = (T1 / Tc) / Kp
            Ti = T1
            Td = 0
        elif (order == 'SO'):
            Kc = ((T1 + T2) / Tc) / Kp
            Ti = T1 + T2
            Td = 0
        elif (order == 'FOPD'):
            Kc = (T1 / (Tc + theta)) / Kp
            Ti = T1
            Td = 0
        elif (order == 'SOPD'):
            Kc = ((T1 + T2) / (Tc + theta)) / Kp
            Ti = T1 + T2
            Td = 0
    elif (control == 'PID'):
        if (order == 'SO'):
            Kc = ((T1 + T2) / Tc) / Kp
            Ti = T1 + T2
            Td = T1 * T2 / (T1 + T2)
        elif (order == 'FOPD'):
            Kc = ((T1 + theta / 2) / (Tc + theta / 2)) / Kp
            Ti = T1 + theta / 2
            Td = T1 * theta / (2 * T1 + theta)
        elif (order == 'SOPD'):
            Kc = ((T1 + T2) / (Tc + theta)) / Kp
            Ti = T1 + T2
            Td = T1 * T2 / (T1 + T2)
    else:
        Kc = 0
        Ti = 0
        Td = 0
    return Kc, Ti, Td

#----------------------------------- 
class Process:
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        self.parameters['Kp'] = parameters['Kp'] if 'Kp' in parameters else 1.0
        self.parameters['theta'] = parameters['theta'] if 'theta' in parameters else 0.0
        self.parameters['Tlead1'] = parameters['Tlead1'] if 'Tlead1' in parameters else 0.0
        self.parameters['Tlead2'] = parameters['Tlead2'] if 'Tlead2' in parameters else 0.0
        self.parameters['Tlag1'] = parameters['Tlag1'] if 'Tlag1' in parameters else 0.0
        self.parameters['Tlag2'] = parameters['Tlag2'] if 'Tlag2' in parameters else 0.0
        self.parameters['Kd'] = parameters['Kd'] if 'Kd' in parameters else 0.0
        self.parameters['Kp_c'] = parameters['Kp_c'] if 'Kp_c' in parameters else 0.0
        self.parameters['Ki'] = parameters['Ki'] if 'Ki' in parameters else 0.0
        self.parameters['T1'] = parameters['T1'] if 'T1' in parameters else 0.0
        self.parameters['T2'] = parameters['T2'] if 'T2' in parameters else 0.0
        
#-----------------------------------        
def Bode_TF(P,omega):
    
    """
    :P: Process as defined by the class "Process".
        Use the following command to define the default process which is simply a unit gain process:
            P = Process({})
        
        A delay and two lag time constants can be added
        
        Use the following commands for a SOPDT process:
            P.parameters['Kp'] = 1.1
            P.parameters['Tlag1'] = 10.0
            P.parameters['Tlag2'] = 2.0
            P.parameters['theta'] = 2.0
        
        Use the following commands for a FOPD process:
            P.parameters['Kp'] = 1.1
            P.parameters['Tlag1'] = 10.0
            P.parameters['theta'] = 2.0
        
    :omega: frequency vector (rad/s); generated by a command of the type "omega = np.logspace(-2, 2, 10000)".
    Ps (P(j omega)) (vector of complex numbers) is returned.
    
    """     
    
    s = 1j*omega
    
    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    PLag1 = 1/(P.parameters['Tlag1']*s + 1)
    PLag2 = 1/(P.parameters['Tlag2']*s + 1)

    
    Ps = np.multiply(Ptheta,PGain)
    Ps = np.multiply(Ps,PLag1)
    Ps = np.multiply(Ps,PLag2)
    return Ps

#-----------------------------------        
def Bode_Ls(P,omega):
    
    """
    :P: Process as defined by the class "Process".
        Use the following command to define the default process which is simply a unit gain process:
            P = Process({})
        
        A delay, 2 process time constants and the PID controller constants can be added.
        
        Use the following commands for the SOPDT process:
            P.parameters['Kp'] = Kp_p
            P.parameters['T1'] = T1_p
            P.parameters['T2'] = T2_p
            P.parameters['theta'] = theta_p

        Use the following commands for the PID Controller process:
            P.parameters['Kd'] = Td
            P.parameters['Kp_c'] = Kc
            P.parameters['Ki'] = Ti      
        
    :omega: frequency vector (rad/s); generated by a command of the type "omega = np.logspace(-5, 2, 70000)".
    
    The function "Bode_Ls" generates the Bode diagram of the process P
    """     
    
    s = 1j*omega
    
    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    P1 = 1/(P.parameters['T1']*s + 1)
    P2 = 1/(P.parameters['T2']*s + 1)
    P3 = 1/s
    P4 = P.parameters['Kd']*s*s + P.parameters['Kp_c']*s + P.parameters['Ki']
    
    Ls = np.multiply(Ptheta,PGain)
    Ls = np.multiply(Ls,P1)
    Ls = np.multiply(Ls,P2)
    Ls = np.multiply(Ls,P3)
    Ls = np.multiply(Ls,P4)
    
    
    
    fig, (ax_gain, ax_phase) = plt.subplots(2,1)
    fig.set_figheight(12)
    fig.set_figwidth(22)

    # Gain part
    ax_gain.semilogx(omega,20*np.log10(np.abs(Ls)),label='L(s)')
    
    # Print components
    
    # ax_gain.semilogx(omega,20*np.log10(np.abs(PGain)),label='Pgain')         
    # ax_gain.semilogx(omega,20*np.log10(np.abs(P3)),label='P3(s)') 
    # if P.parameters['theta'] > 0:
    #     ax_gain.semilogx(omega,20*np.log10(np.abs(Ptheta)),label='Ptheta(s)')
    # if P.parameters['T1'] > 0:
    #     ax_gain.semilogx(omega,20*np.log10(np.abs(P1)),label='P1(s)')
    # if P.parameters['T2'] > 0:        
    #     ax_gain.semilogx(omega,20*np.log10(np.abs(P2)),label='P2(s)')       
    # if (P.parameters['Kd'] > 0 or P.parameters['Kp_c'] > 0 or P.parameters['Ki'] > 0):    
    #     ax_gain.semilogx(omega,20*np.log10(np.abs(P4)),label='P4')
    gain_min = np.min(20*np.log10(np.abs(Ls)/5))
    gain_max = np.max(20*np.log10(np.abs(Ls)*5))
    ax_gain.set_xlim([np.min(omega), np.max(omega)])
    ax_gain.set_ylim([gain_min, gain_max])
    ax_gain.set_ylabel('Amplitude |P| [db]')
    ax_gain.set_title('Bode plot of P')
    ax_gain.legend(loc='best')


    ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ls)),label='L(s)')
    
    # Print components
    
    # ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(PGain)),label='Pgain')
    # ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(P3)),label='P3(s)')
    # if P.parameters['theta'] > 0:    
    #     ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ptheta)),label='Ptheta(s)')
    # if P.parameters['T1'] > 0:        
    #     ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(P1)),label='P1(s)')
    # if P.parameters['T2'] > 0:        
    #     ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(P2)),label='P2(s)')
    # if (P.parameters['Kd'] > 0 or P.parameters['Kp_c'] > 0 or P.parameters['Ki'] > 0):       
    #     ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(P4)),label='P4')
    
    ax_phase.set_xlim([np.min(omega), np.max(omega)])
    ph_min = np.min((180/np.pi)*np.unwrap(np.angle(Ls))) - 10
    ph_max = np.max((180/np.pi)*np.unwrap(np.angle(Ls))) + 10
    ax_phase.set_ylim([np.max([ph_min, -360]), ph_max])
    ax_phase.set_ylabel(r'Phase $\angle P$ [°]')
    ax_phase.legend(loc='best')
    
    
    #-----------------------------------        
def Stability_Margins(P,omega):
    """
    :P: Process as defined by the class "Process".
    :omega: frequency vector (rad/s); generated by a command of the type "omega = np.logspace(-5, 2, 70000)". 
    
    The function "Stability_Margins" returns the gain and phase margins of the L(s) transfer function. If one of the margins is infinite, returns inf
    """
    s = 1j*omega
    
    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    P1 = 1/(P.parameters['T1']*s + 1)
    P2 = 1/(P.parameters['T2']*s + 1)
    P3 = 1/s
    P4 = P.parameters['Kd']*s*s + P.parameters['Kp_c']*s + P.parameters['Ki']
    
    Ls = np.multiply(Ptheta,PGain)
    Ls = np.multiply(Ls,P1)
    Ls = np.multiply(Ls,P2)
    Ls = np.multiply(Ls,P3)
    Ls = np.multiply(Ls,P4)
    
    
    phase_crossover = -1
    gain_crossover = -1
    test_gain = 20*np.log10(Ls).real
    test_phase = (180/np.pi)*np.unwrap(np.angle(Ls))


    if (test_gain[1]>0 and test_phase[1]>-180):    
        for i in range (0,70000):
            if (test_gain[i] > 0.00000000001):
                gain_crossover = i                
            if (test_phase[i] > -179.99999999):
                phase_crossover = i
                
    if (test_gain[1]>0 and test_phase[1]<-180):
        for i in range (0,70000):
            if (test_gain[i] > 0.00000000001):
                gain_crossover = i                
            if (test_phase[i] < -180.000001):
                phase_crossover = i   
                
    if (test_gain[1]<0 and test_phase[1]>-180):
        for i in range (0,70000):
            if (test_gain[i] < -0.00000000001):
                gain_crossover = i                
            if (test_phase[i] > -179.99999999):
                phase_crossover = i
                
    if (test_gain[1]<0 and test_phase[1]<-180):
        for i in range (0,70000):
            if (test_gain[i] < -0.00000000001):
                gain_crossover = i                
            if (test_phase[i] < -180.000001):
                phase_crossover = i
                
        
    if (test_phase[gain_crossover] < -180):        
        phase_margin = test_phase[gain_crossover] + 180
    else:
        phase_margin = -1*test_phase[gain_crossover] + 180
        
    if (test_gain[phase_crossover] > 0):
        gain_margin = -1*test_gain[phase_crossover]
    else:
        gain_margin = np.abs(test_gain[phase_crossover])
        
    
    inf = float('inf')
    
    if (gain_crossover == 69999 or phase_crossover == 69999):
        gain_margin = inf
        phase_margin = inf       
    
    
    return gain_margin, phase_margin