import torch


class gait_scheduler():
    def __init__(self,cfg_gait,num_envs,device,dt):
        self.num_envs = num_envs
        self.device = device
        
        self.phase_offsets = cfg_gait.phase_offsets
        self.switchingPhaseNominal = cfg_gait.switchingPhaseNominal
        self.nom_gait_period = cfg_gait.nom_gait_period
        self.num_legs = len(self.phase_offsets)
        self.dt = dt

        self._init_gait_scheduler()

    def _init_gait_scheduler(self):

        # * init buffer for phase main variable
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device, requires_grad=False)

        # * init buffer for individual leg phase variable
        
        # self.LegPhase = torch.hstack((self.phase_offsets[0]*torch.ones(self.num_envs, 1, dtype=torch.float,
        #                          device=self.device, requires_grad=False),\
        #                                self.phase_offsets[1]*torch.ones(self.num_envs, 1, dtype=torch.float,
        #                          device=self.device, requires_grad=False)))

        # self.LegPhaseStance = torch.hstack((self.phase_offsets[0]*torch.ones(self.num_envs, 1, dtype=torch.float,
        #                          device=self.device, requires_grad=False),\
        #                                self.phase_offsets[1]*torch.ones(self.num_envs, 1, dtype=torch.float,
        #                          device=self.device, requires_grad=False)))

        # self.LegPhaseSwing = torch.hstack((self.phase_offsets[0]*torch.ones(self.num_envs, 1, dtype=torch.float,
        #                          device=self.device, requires_grad=False),\
        #                                self.phase_offsets[1]*torch.ones(self.num_envs, 1, dtype=torch.float,
        #                          device=self.device, requires_grad=False)))
        

        self.LegPhase = torch.tensor([],device=self.device, requires_grad=False)
        self.LegPhaseStance = torch.tensor([],device=self.device, requires_grad=False)
        self.LegPhaseSwing = torch.tensor([],device=self.device, requires_grad=False)

        for i in range(self.num_legs):
            print('i = '+str(i))
            #phase_offsets_ = phase_offsets_.append(torch.tensor([self.phase_offsets[i]]).expand(num_env,1))
            self.LegPhase = torch.hstack((self.LegPhase,self.phase_offsets[i]*torch.ones(self.num_envs, 1, dtype=torch.float,
                                    device=self.device, requires_grad=False)))

            self.LegPhaseStance = torch.hstack((self.LegPhaseStance,self.phase_offsets[i]*torch.ones(self.num_envs, 1, dtype=torch.float,
                                    device=self.device, requires_grad=False)))

            self.LegPhaseSwing = torch.hstack((self.LegPhaseSwing,self.phase_offsets[i]*torch.ones(self.num_envs, 1, dtype=torch.float,
                                    device=self.device, requires_grad=False)))
            
        A=10
        print("v1 = "+ str(self.LegPhase))
        
        #print(self.LegPhaseSwing)


    def increment_phase(self):
        """ Callback called before computing terminations, rewards, and observations, phase-dynamics
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # increment phase variable
        dphase = self.dt / self.nom_gait_period

        self.phase = torch.fmod(self.phase + dphase, 1)

        for foot in range(self.num_legs):
            self.LegPhase[:, foot] = torch.fmod(self.LegPhase[:, foot] + dphase, 1)

            in_stance_bool = (self.LegPhase[:, foot] <= self.switchingPhaseNominal)
            in_swing_bool = torch.logical_not(in_stance_bool)

            self.LegPhaseStance[:, foot] = self.LegPhase[:, foot] / self.switchingPhaseNominal*in_stance_bool +\
                                           in_swing_bool*1 # Stance phase has completed since foot is in swing

            self.LegPhaseSwing[:, foot] = 0 * in_stance_bool \
                                          + in_swing_bool*(self.LegPhase[:, foot] - self.switchingPhaseNominal)\
                                          / (1.0 - self.switchingPhaseNominal)

        #     if (self.LegPhase[:, foot] <= self.cfg.gait.switchingPhaseNominal):
        #         self.LegPhaseStance[:, foot] = self.LegPhase[:, foot]/self.cfg.gait.switchingPhaseNominal
        #         self.LegPhaseSwing[:, foot] = 0  # Swing phase has not started since foot is in stance

        #     # in swing phase
        #     else:
        #         self.LegPhaseStance[:, foot] = 1.  # Stance phase has completed since foot is in swing
        #         self.LegPhaseSwing[:, foot] = (self.LegPhase[:, foot] - self.cfg.gait.switchingPhaseNominal)\
        #                                       / (1.0 - self.cfg.gait.switchingPhaseNominal)

if __name__== "__main__":
    # script for testing the gait scheduler
    import math as mt
    import matplotlib.pyplot as plt
    import numpy as np
    
    # gait config class needed to initialize the gait scheduler
    class gait():
        nom_gait_period = 0.8
        phase_offsets = [0, 0.5, 0.1] # phase offset for each leg the length of this vector also determined the nb of legs used for the gait schedule
        switchingPhaseNominal = 0.5 # switch phase from stance to swing
    
    # environment step dt
    dt = 0.002
    cfg_gait = gait()


    num_env = 2
    GS = gait_scheduler(cfg_gait,num_env,'cpu',dt)

    Ttot = 2
    N_iter =  mt.floor(Ttot/dt)
    numlegs = len(cfg_gait.phase_offsets)
    main_phase = np.zeros([num_env, N_iter])
    leg_phase = np.zeros([num_env,numlegs, N_iter])
    stance_phase = np.zeros([num_env,numlegs, N_iter])
    swing_phase = np.zeros([num_env,numlegs, N_iter])
    
    time = (np.arange(0,N_iter)*dt)
    
    for i in range(N_iter):
        GS.increment_phase()

        main_phase[:,i]  = GS.phase.detach().cpu().numpy().ravel()
        leg_phase[:,:,i] = GS.LegPhase.detach().cpu().numpy()
        stance_phase[:,:,i] = GS.LegPhaseStance.detach().cpu().numpy()
        swing_phase[:,:,i] = GS.LegPhaseSwing.detach().cpu().numpy()

    
    nrows = 5
    ncols = 1
    idx=1
    
    plt.figure()
    plt.subplot(nrows, ncols, idx);idx+=1
    plt.plot(time, main_phase.T)
    plt.title('main phase')

    plt.subplot(nrows, ncols, idx);idx+=1
    for i in range(numlegs):
        plt.plot(time,leg_phase[:,i,:].T)
    plt.title('leg phase')
    #plt.legend(["leg0","leg1"])

    plt.subplot(nrows, ncols, idx);idx+=1
    for i in range(numlegs):
       plt.plot(time,stance_phase[:,i,:].T)
    plt.title('stance phase')
    plt.legend(["leg0","leg1"])

    plt.subplot(nrows, ncols, idx);idx+=1
    for i in range(numlegs):
        plt.plot(time,swing_phase[:,i,:].T)
    plt.title('swing phase')
    plt.legend(["leg0","leg1"])
    plt.show()