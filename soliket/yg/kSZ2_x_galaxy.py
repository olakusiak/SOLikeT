"""
.. module:: yxg

:Synopsis: Definition y-galaxy power spectrum likelihood .

:running boris: $ /usr/local/anaconda3/bin/mpirun -np 4 /usr/local/anaconda3/bin/cobaya-run soliket/ymap/input_files/yxg_ps.yaml -f
:running ola: $ /Users/boris/opt/anaconda3/bin/mpirun -np 4 /Users/boris/opt/anaconda3/bin/cobaya-run soliket/ymap/input_files/yxg_ps_template.yaml -f

"""


from cobaya.theory import Theory
# from cobaya.conventions import _packages_path
# from cobaya.likelihoods._base_classes import _InstallableLikelihood
from soliket.gaussian import GaussianLikelihood
import numpy as np
import os
from scipy.ndimage.interpolation import shift
from typing import Optional, Sequence
from pkg_resources import resource_filename
from scipy.interpolate import interp1d



class KSZ2XG_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    data_file: Optional[str] = None
    # cov_data_file: Optional[str] = None
    bp_wind_yg_file: Optional[str] = None
    Nbins: Optional[str] = None

    # Load the data
    def initialize(self):
        self.datafile = self.data_file
        self.bpwf = np.load(os.path.join(self.data_directory, self.bp_wind_yg_file))[0]
        Npoints = self.Nbins

        D = np.loadtxt(os.path.join(self.data_directory, self.data_file))
        # cov = np.loadtxt(os.path.join(self.data_directory, self.cov_data_file))

        self.ell = D[0,:Npoints]
        self.yg = D[1,:Npoints]
        self.sigma_tot = D[2,:Npoints]
        self.covmat =   np.diag(self.sigma_tot**2)
        print("ell ola:", self.ell)
        print("yg ola:", self.yg)
        print("sigma ola:", self.sigma_tot**2)
        print("cov :", np.diag(self.covmat))

        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        super().initialize()


    def get_requirements(self):
        return {"Cl_kSZ_kSZ_g": {},}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell
        y_data = self.yg
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov

    def _bin(self, ell_theory, cl_theory, ell_data, ellmax, bpwf, pix_win, Nellbins=31, conv2cl=True,):
        """
        Interpolate the theory dl's, and bin according to the bandpower window function (bpwf)
        """
        #interpolate
        new_ell = np.arange(2, ellmax, 1)
        cl_theory_log = np.log(cl_theory)
        f_int =  interp1d(ell_theory, cl_theory_log, fill_value="extrapolate")
        inter_cl_log = np.asarray(f_int(new_ell))
        inter_cl= np.exp(inter_cl_log)
        if conv2cl==True: #go from dls to cls because the bpwf mutliplies by ell*(ell+1)/2pi
            inter_cl= inter_cl*(2.0*np.pi)/(new_ell)/(new_ell+1.0)

        #multiply by the pixel window function (from healpix for given nside)
        inter_cl = inter_cl*(pix_win[2:ellmax])**2
        #bin according to the bpwf
        cl_binned = np.zeros(Nellbins)
        for i in range (Nellbins):
            wi = bpwf[i]
            # wi starts from ell=2 according to Alex, email 1-9-22; could add ell=0,1, but would contribute nothing to the sum
            cl_binned[i] = np.sum(wi[2:len(inter_cl)+2]*inter_cl)
        #print("clbinned:", cl_binned)
        return ell_data, cl_binned

    def _get_theory(self, **params_values_dict):
        Aksz=params_values_dict['A_ksz']
        Agal=params_values_dict['A_gal']
        bpwf=self.bpwf[:,0,:]
        Npoints = self.Nbins
        ellmax_bin = 5600

        # ########
        # Cl_yxg
        ########
        theory_yg = self.provider.get_Cl_kSZ_kSZ_g()
        ell_theory_yg = theory_yg['ell']
        cl_1h_theory_yg = theory_yg['1h']
        cl_2h_theory_yg = theory_yg['2h']
        cl_lensing = theory_yg['lensing term']

        print("cl_1h_theory_yg:", cl_1h_theory_yg[:10])
        print("ell_theory_yg:", ell_theory_yg[:10])
        print("cl_2h_theory_yg:", cl_2h_theory_yg[:10])
        print("cl_lensing:", cl_lensing[:10])
        # dl_theory_yg = np.asarray(list(cl_1h_theory_yg)) + np.asarray(list(cl_2h_theory_yg))
        # ell_yg_bin, dl_yg_bin = self._bin(ell_theory_yg, dl_theory_yg, self.ell, ellmax_bin, bpwf, pixwin, Nellbins=Npoints, conv2cl=True)
        # #print("yg bin: ", dl_yg_bin[:10])

       
        # cl_tot = Aksz*Ag*(dl_yg_bin ) + 2*(alpha-1)*dl_ym_bin)
        # if np.isnan(cl_tot).any()==True:
        #     print("Nans in the theory prediction!")
        #     exit()

        return Aksz* Agal* cl_2h_theory_yg 
