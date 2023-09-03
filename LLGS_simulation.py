import numpy as np
from pathlib import Path
from numba import jit, njit, prange, set_num_threads
from lattice import lattice_2D,lattice_3D
from sklearn.preprocessing import normalize
set_num_threads = 8

@njit(parallel=True)
def calculate_spin_velocities_jit(N,H_E,H_v,H_t,H_DMI_x,H_DMI_y,H_DMI_z,H_ext,H_FL,H_DL,alpha,spins,svels):
    gyro_magnetic_ratio = -1.76085963023E2
    next_svels = np.zeros((N,3))

    H_eff = np.zeros((N,3))
    # calculate H_eff
    #---------------------------------------------------------
    # spin-spin individual part: H_v H_t H_FL H_ext
    #---------------------------------------------------------
    for i in prange(N):
        H_eff[i,:] += H_ext+H_FL+H_t*np.array([2*spins[i,0],0,0])-H_v*np.array([0,0,2*spins[i,2]])
    #---------------------------------------------------------
    # spin-spin interactive part: H_E H_DMI
    #---------------------------------------------------------
    def interactive_energy(H_E,H_DMI_x,H_DMI_y,H_DMI_z,spins):
        sx = spins[:,0]
        sy = spins[:,1]
        sz = spins[:,2]

        sx = np.ascontiguousarray(sx)
        sy = np.ascontiguousarray(sy)
        sz = np.ascontiguousarray(sz)

        #exchange energy
        E_ex = np.dot(sx,np.dot(H_E,sx)) + np.dot(sy,np.dot(H_E,sy)) +np.dot(sz,np.dot(H_E,sz))
        #DMI energy
        E_DMI_x = np.dot(sy,np.dot(H_DMI_x,sz)) - np.dot(sz,np.dot(H_DMI_x,sy))
        E_DMI_y = np.dot(sz,np.dot(H_DMI_y,sx)) - np.dot(sx,np.dot(H_DMI_y,sz))
        E_DMI_z = np.dot(sx,np.dot(H_DMI_z,sy)) - np.dot(sy,np.dot(H_DMI_z,sx))
        
        return E_ex + E_DMI_x + E_DMI_y + E_DMI_z

    epsilon = 1E-8
    for i in prange(N):
        for j in prange(3):
            epsilon_spin = np.zeros((N,3))
            epsilon_spin[i,j] = epsilon
            H_eff[i,j] += (interactive_energy(H_E,H_DMI_x,H_DMI_y,H_DMI_z,spins)-interactive_energy(H_E,H_DMI_x,H_DMI_y,H_DMI_z,spins+epsilon_spin))/epsilon
    #---------------------------------------------------------
    #calculate spin velocities
    for i in prange(N):
        next_svels[i,:] = gyro_magnetic_ratio*(np.cross(H_eff[i,:],spins[i,:])+np.cross(np.cross(H_DL,spins[i,:]),spins[i,:]))+alpha*np.cross(spins[i,:],svels[i,:])

    return next_svels


class LLGS_Simulation_2D:

    def __init__(self, lattice:lattice_2D) :
        self.lattice = lattice
        self.N = lattice.N
        self.time = 0
        self.setup()
        self._H_E = np.zeros((self.N,self.N))
        self._H_DMI_x = np.zeros((self.N,self.N))
        self._H_DMI_y = np.zeros((self.N,self.N))
        self._H_DMI_z = np.zeros((self.N,self.N))
        return
    
    def setup(self, H_v=0,
                    H_t=0,
                    H_ext=np.zeros(3),
                    H_FL=np.zeros(3),
                    H_DL=np.zeros(3),
                    alpha=0,
                    method='RK4',
                    io_freq=100, 
                    io_title="lattice",
                    io_screen=False,
                    ):
        """
        param:
        ----------------------------------------------
        H_E: (N,N) exchange field
        H_v: perpendicular anisotropy
        H_t: inplane anisotropy
        H_DMI_x: (N,N) x component of DMI field
        H_DMI_y: (N,N) y component of DMI field
        H_DMI_z: (N,N) z component of DMI field
        H_ext: (3) external field
        H_FL: (3) field like SOT (current direction)
        H_DL: (3) damped like SOT (current direction)
        alpha: Gilbert damping constant
        method: string, the numerical scheme, support 'Euler','RK2','RK4'
        io_freq: int, the frequency to outupt data.
        io_title: the output header
        io_screen: print message on screen or not
        ----------------------------------------------
        """
        self._H_v = H_v
        self._H_t = H_t
        self._H_ext = H_ext
        self._H_FL = H_FL
        self._H_DL = H_DL
        self._alpha = alpha
        self.method = method
        self._io_freq = io_freq 
        self._io_title = io_title
        self._io_screen = io_screen
        return
    
    #setter
    def set_exchange_field(self,H_E):
        self._H_E = H_E
        return
    def set_DMI_field_x(self,H_DMI_X):
        self._H_DMI_x = H_DMI_X
        return
    def set_DMI_field_y(self,H_DMI_Y):
        self._H_DMI_y = H_DMI_Y
        return
    def set_DMI_field_z(self,H_DMI_Z):
        self._H_DMI_z = H_DMI_Z
        return
    def set_H_ext(self,H_ext):
        self._H_ext = H_ext
        return
    def set_H_FL(self,H_FL):
        self._H_FL = H_FL
        return
    def set_H_DL(self,H_DL):
        self._H_DL = H_DL
        return
    
    #getter
    def get_exchange_field(self):    
        return self._H_E
    def get_DMI_field_x(self):    
        return self._H_DMI_x
    def get_DMI_field_y(self):    
        return self._H_DMI_y
    def get_DMI_field_z(self):    
        return self._H_DMI_z
    def get_H_ext(self):   
        return self._H_ext
    def get_H_FL(self):       
        return self._H_FL
    def get_H_DL(self):
        return self._H_DL
    
    
    
    
    
    def evolve(self, lattice:lattice_2D, dt:float=0.001, tmax:float=0.1):
        """
        param:
        -----------------------------------------------
        dt: time step
        tmax: final time
        -----------------------------------------------
        """
        method = self.method
        if method=="Euler":
            _update_spin = self._update_spin_euler
        elif method=="RK2":
            _update_spin = self._update_spin_rk2
        elif method=="RK4":
            _update_spin = self._update_spin_rk4    
        else:
            raise ValueError("'method' must be 'Euler','RK2','RK4'")
        
        io_folder_name = "data_"+self._io_title
        Path(io_folder_name).mkdir(parents=True, exist_ok=True)

        time=self.time
        nsteps=np.ceil((tmax-time)/dt)

        fn_structure = io_folder_name + "/structure.txt"
        lattice.output_lattice_structure(fn_structure)

        for n in range(int(nsteps)):

            fn=io_folder_name+"/"+self.method+"_"+str(n).zfill(5)+".txt"
            if (time+dt>tmax): dt=tmax-time
            if (n % self._io_freq == 0) and self._io_screen:
                print("fn =",fn,"time =",time,"dt=",dt)
            
            lattice.output_spin_data(fn,time)
            _update_spin(lattice,dt)
            time+=dt

        print("simulation is done")
        return
    
    def _update_spin_euler(self,lattice:lattice_2D,dt):

        spins = lattice.get_spins()
        svels = lattice.get_spin_velocities()

        N = lattice.N
        H_E = self._H_E
        H_v = self._H_v
        H_t = self._H_t
        H_DMI_x = self._H_DMI_x
        H_DMI_y = self._H_DMI_y
        H_DMI_z = self._H_DMI_z
        H_ext = self._H_ext
        H_FL = self._H_FL
        H_DL = self._H_DL
        alpha = self._alpha


        next_svels = calculate_spin_velocities_jit( N=N,
                                                    H_E=H_E,
                                                    H_v=H_v,
                                                    H_t=H_t,
                                                    H_DMI_x=H_DMI_x,
                                                    H_DMI_y=H_DMI_y,
                                                    H_DMI_z=H_DMI_z,
                                                    H_ext=H_ext,
                                                    H_FL=H_FL,
                                                    H_DL=H_DL,
                                                    alpha=alpha,
                                                    spins=spins,
                                                    svels=svels
                                                    )
        next_spins = spins + dt*svels
        next_spins = normalize(next_spins)

        lattice.set_spins(next_spins)
        lattice.set_spin_velocities(next_svels)
        return lattice
    
    def _update_spin_rk2(self,lattice:lattice_2D,dt):

        #k1
        spins = lattice.get_spins()
        svels = lattice.get_spin_velocities()

        N = lattice.N
        H_E = self._H_E
        H_v = self._H_v
        H_t = self._H_t
        H_DMI_x = self._H_DMI_x
        H_DMI_y = self._H_DMI_y
        H_DMI_z = self._H_DMI_z
        H_ext = self._H_ext
        H_FL = self._H_FL
        H_DL = self._H_DL
        alpha = self._alpha

        #k2
        spin2 = spins + dt/2*svels
        svel2 = calculate_spin_velocities_jit( N=N,
                                                    H_E=H_E,
                                                    H_v=H_v,
                                                    H_t=H_t,
                                                    H_DMI_x=H_DMI_x,
                                                    H_DMI_y=H_DMI_y,
                                                    H_DMI_z=H_DMI_z,
                                                    H_ext=H_ext,
                                                    H_FL=H_FL,
                                                    H_DL=H_DL,
                                                    alpha=alpha,
                                                    spins=spin2,
                                                    svels=svels
                                                    )
        next_spins = spins + dt/2*(svels+svel2)
        next_spins = normalize(next_spins)

        next_svels = calculate_spin_velocities_jit( N=N,
                                                    H_E=H_E,
                                                    H_v=H_v,
                                                    H_t=H_t,
                                                    H_DMI_x=H_DMI_x,
                                                    H_DMI_y=H_DMI_y,
                                                    H_DMI_z=H_DMI_z,
                                                    H_ext=H_ext,
                                                    H_FL=H_FL,
                                                    H_DL=H_DL,
                                                    alpha=alpha,
                                                    spins=next_spins,
                                                    svels=svels
                                                    )

        lattice.set_spins(next_spins)
        lattice.set_spin_velocities(next_svels)
        return lattice

    def _update_spin_rk4(self,lattice:lattice_2D,dt):

            #k1
            spins = lattice.get_spins()
            svels = lattice.get_spin_velocities()

            N = lattice.N
            H_E = self._H_E
            H_v = self._H_v
            H_t = self._H_t
            H_DMI_x = self._H_DMI_x
            H_DMI_y = self._H_DMI_y
            H_DMI_z = self._H_DMI_z
            H_ext = self._H_ext
            H_FL = self._H_FL
            H_DL = self._H_DL
            alpha = self._alpha

            #k2
            spin2 = spins + dt/2*svels
            svel2 = calculate_spin_velocities_jit( N=N,
                                                        H_E=H_E,
                                                        H_v=H_v,
                                                        H_t=H_t,
                                                        H_DMI_x=H_DMI_x,
                                                        H_DMI_y=H_DMI_y,
                                                        H_DMI_z=H_DMI_z,
                                                        H_ext=H_ext,
                                                        H_FL=H_FL,
                                                        H_DL=H_DL,
                                                        alpha=alpha,
                                                        spins=spin2,
                                                        svels=svels
                                                        )
            
            #k3
            spin3 = spins + dt/2*svel2
            svel3 = calculate_spin_velocities_jit( N=N,
                                                        H_E=H_E,
                                                        H_v=H_v,
                                                        H_t=H_t,
                                                        H_DMI_x=H_DMI_x,
                                                        H_DMI_y=H_DMI_y,
                                                        H_DMI_z=H_DMI_z,
                                                        H_ext=H_ext,
                                                        H_FL=H_FL,
                                                        H_DL=H_DL,
                                                        alpha=alpha,
                                                        spins=spin3,
                                                        svels=svels
                                                        )
            
            #k4
            spin4 = spins + dt*svels
            svel4 = calculate_spin_velocities_jit( N=N,
                                                        H_E=H_E,
                                                        H_v=H_v,
                                                        H_t=H_t,
                                                        H_DMI_x=H_DMI_x,
                                                        H_DMI_y=H_DMI_y,
                                                        H_DMI_z=H_DMI_z,
                                                        H_ext=H_ext,
                                                        H_FL=H_FL,
                                                        H_DL=H_DL,
                                                        alpha=alpha,
                                                        spins=spin4,
                                                        svels=svels
                                                        )


            next_spins = spins + dt/6*(svels+2*svel2+2*svel3+svel4)
            next_spins = normalize(next_spins)

            next_svels = calculate_spin_velocities_jit( N=N,
                                                        H_E=H_E,
                                                        H_v=H_v,
                                                        H_t=H_t,
                                                        H_DMI_x=H_DMI_x,
                                                        H_DMI_y=H_DMI_y,
                                                        H_DMI_z=H_DMI_z,
                                                        H_ext=H_ext,
                                                        H_FL=H_FL,
                                                        H_DL=H_DL,
                                                        alpha=alpha,
                                                        spins=next_spins,
                                                        svels=svels
                                                        )

            lattice.set_spins(next_spins)
            lattice.set_spin_velocities(next_svels)
            return lattice