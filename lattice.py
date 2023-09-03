import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class lattice_3D:

    def __init__(self,n_site,n_a,n_b,n_c):
        '''
        param: 
        ------------------------------------------------
        _n_site: number of particle in one Bragg basis
        _n_a: number of Bragg basis in a-axis 
        _n_b: number of Bragg basis in b-axis 
        _n_c: number of Bragg basis in c-axis
        -------------------------------------------------
        by default:

        _N: number of particle
        _tags:  (N,4) index of each particle, columns: site,a,b,c
        _position: (N,3) for each particle
        _spins: (N,3) for each particle
        _spin_volocities: (N,3) for each particle
        '''
        self._n_site = n_site
        self._n_a = n_a
        self._n_b = n_b
        self._n_c = n_c
        N = n_site*n_a*n_b*n_c
        self.N = N
        self._tag = np.zeros((N,4))
        self._positions=np.zeros((N,3))
        self._spins=np.zeros((N,3))
        self._spin_velocities=np.zeros((N,3))
        for a in range(n_a):
            for b in range(n_b):
                for c in range(n_c):
                    for s in range(n_site):
                        n = s + a*n_site + b*n_site*n_a +c*n_site*n_a*n_b #index of particle
                        self._tag[n,0] = s
                        self._tag[n,1] = a
                        self._tag[n,2] = b
                        self._tag[n,3] = c
        self._tag.astype(np.int32)

        return


    
    #getter
    def get_tags(self):
        return self._tag
    def get_positions(self): 
        return self._positions
    def get_spins(self):
        return self._spins
    def get_spin_velocities(self):
        return self._spin_velocities

    #setter
    def set_spins(self,spins):
        self._spins = spins
        return 
    def set_spin_velocities(self,vel):
        self._spin_velocities = vel
        return
    
    def output_lattice_structure(self,fn):
        """
        Write lattice data into a file named "fn"
        """
        tag = self._tag
        pos = self._positions

        header = """
                ----------------------------------------------------
                rows : i-particle; 
                columns  : site, a, b, c, x, y, z
                ----------------------------------------------------
                """
        np.savetxt(fn,np.transpose((tag[:,0], tag[:,1], tag[:,2], tag[:,3], pos[:,0], pos[:,1], pos[:,2] )),header=header)

        return
    
    def output_spin_data(self,fn, time):
        """
        Write spin simulation data into a file named "fn"
        """
        s = self._spins

        header = """
                ----------------------------------------------------
                rows : i-particle; 
                columns  : sx, sy, sz
                ----------------------------------------------------
                """
        header += "time = {}".format(time)
        np.savetxt(fn,np.transpose((s[:,0],s[:,1],s[:,2])),header=header)

        return
    
    def set_position(self, r_a, r_b, r_c, r_site): 
        """
        ----------------------------------------
        set the position of each particle
        r_a: (3) basis vector on a axis for x, y, z coordinate
        r_b: (3) basis vector on b axis for x, y, z coordinate 
        r_c: (3) basis vector on c axis for x, y, z coordinate 
        r_site: (n_site,3) x,y,z coordinate for each site
        ----------------------------------------
        example:
        r_site = np.array(([r1],[r2])) r1: site 1, r2: site 2
        ----------------------------------------
        """
        n_site = self._n_site
        n_a = self._n_a 
        n_b = self._n_b 
        n_c = self._n_c
        for a in range(n_a):
            for b in range(n_b):
                for c in range(n_c):
                    for s in range(n_site):
                        if n_site == 1:
                            n = s + a*n_site + b*n_site*n_a +c*n_site*n_a*n_b #index of particle
                            self._positions[n,0] = a*r_a[0] + b*r_b[0] + c*r_c[0] + r_site[0]
                            self._positions[n,1] = a*r_a[1] + b*r_b[1] + c*r_c[1] + r_site[1]
                            self._positions[n,2] = a*r_a[2] + b*r_b[2] + c*r_c[2] + r_site[2]
                        else:
                            n = s + a*n_site + b*n_site*n_a +c*n_site*n_a*n_b #index of particle
                            self._positions[n,0] = a*r_a[0] + b*r_b[0] + c*r_c[0] + r_site[s,0]
                            self._positions[n,1] = a*r_a[1] + b*r_b[1] + c*r_c[1] + r_site[s,1]
                            self._positions[n,2] = a*r_a[2] + b*r_b[2] + c*r_c[2] + r_site[s,2]
        return

class lattice_2D:

    def __init__(self,n_site,n_a,n_b):
        '''
        param 
        ------------------------------------------------
        _n_site: number of particle in one Bragg basis
        _n_a: number of Bragg basis in a-axis 
        _n_b: number of Bragg basis in b-axis 
        -------------------------------------------------
        by default:

        N: number of particle
        _tags:  (N,3) index of each particle, columns: site,a,b
        _position: (N,2) for each particle
        _spins: (N,3) for each particle
        _spin_volocities: (N,3) for each particle
        '''
        self._n_site = n_site
        self._n_a = n_a
        self._n_b = n_b
        N = n_site*n_a*n_b
        self.N = N
        self._tag = np.zeros((N,3))
        self._positions=np.zeros((N,2))
        self._spins=np.zeros((N,3))
        self._spin_velocities=np.zeros((N,3))
        for a in range(n_a):
            for b in range(n_b):
                for s in range(n_site):
                    n = s + a*n_site + b*n_site*n_a #index of particle
                    self._tag[n,0] = s
                    self._tag[n,1] = a
                    self._tag[n,2] = b
        self._tag.astype(np.int32)
        return



    
    #getter
    def get_tags(self):
        return self._tag
    def get_positions(self): 
        return self._positions
    def get_spins(self):
        return self._spins
    def get_spin_velocities(self):
        return self._spin_velocities

    #setter
    def set_spins(self,spins):
        self._spins = spins
        return 
    def set_spin_velocities(self,vel):
        self._spin_velocities = vel
        return
    
    def output_lattice_structure(self,fn):
        """
        Write lattice data into a file named "fn"
        """
        tag = self._tag
        pos = self._positions

        header = """
                ----------------------------------------------------
                rows : i-particle; 
                columns  : site, a, b, x, y
                ----------------------------------------------------
                """
        np.savetxt(fn,np.transpose((tag[:,0], tag[:,1], tag[:,2], pos[:,0], pos[:,1] )),header=header)

        return
    
    def output_spin_data(self,fn, time):
        """
        Write spin simulation data into a file named "fn"
        """
        s = self._spins

        header = """
                ----------------------------------------------------
                rows : i-particle; 
                columns  : sx, sy, sz
                ----------------------------------------------------
                """
        header += "time = {}".format(time)
        np.savetxt(fn,np.transpose((s[:,0],s[:,1],s[:,2])),header=header)

        return
    
    def set_position(self, r_a, r_b, r_site): 
        """
        ----------------------------------------
        set the position of each particle
        r_a: (2) basis vector on a axis for x, y coordinate
        r_b: (2) basis vector on b axis for x, y coordinate 
        r_site: (n_site,2) x,y coordinate for each site
        ----------------------------------------
        example:
        r_site = np.array(([r1],[r2])) r1: site 1, r2: site 2
        ----------------------------------------
        """
        n_site = self._n_site
        n_a = self._n_a 
        n_b = self._n_b 
        for a in range(n_a):
            for b in range(n_b):
                for s in range(n_site):
                    if n_site == 1:
                        n = a*n_site + b*n_site*n_a #index of particle
                        self._positions[n,0] = a*r_a[0] + b*r_b[0] + r_site[0]
                        self._positions[n,1] = a*r_a[1] + b*r_b[1] + r_site[1]
                    else:
                        n = s + a*n_site + b*n_site*n_a #index of particle
                        self._positions[n,0] = a*r_a[0] + b*r_b[0] + r_site[s,0]
                        self._positions[n,1] = a*r_a[1] + b*r_b[1] + r_site[s,1]
        return

    def plot(self,title=None,arrowscale=0.3,fn=None):
        x = self.get_positions()[:,0]
        y = self.get_positions()[:,1]
        sx = self.get_spins()[:,0]
        sy = self.get_spins()[:,1]
        sz = self.get_spins()[:,2]

        plt.style.use("dark_background")
        plt.set_cmap('bwr')
        fig, ax =plt.subplots()
        cmap = plt.get_cmap("bwr")
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm,ax=ax)
        ax.plot(x,y,'w.',markersize=2)
        if np.linalg.norm(sx)==0 and np.linalg.norm(sy)==0:
            ax.scatter(x,y,c=sz,norm=norm)
        else:    
            ax.quiver(x,y,sx*arrowscale,sy*arrowscale,sz,norm=norm)
        ax.set_aspect('equal')
        ax.axis("off")
        ax.set_title(title)
        if fn != None:
            fig.savefig(fn)
        return














