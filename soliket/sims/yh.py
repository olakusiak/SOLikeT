"""
.. module:: kxg

:Synopsis: Definition y-galaxy power spectrum likelihood .

:running boris: $ /usr/local/anaconda3/bin/mpirun -np 4 /usr/local/anaconda3/bin/cobaya-run soliket/ymap/input_files/kxg_ps.yaml -f
:running ola: $ /Users/boris/opt/anaconda3/bin/mpirun -np 4 /Users/boris/opt/anaconda3/bin/cobaya-run soliket/ymap/input_files/kxg_ps_template.yaml -f

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
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.special import jv
from scipy.integrate import simps
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import scipy
print(scipy.__version__)

class YXHALOS_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    yxh_data_file: Optional[str] = None
    cov_data_file: Optional[str] = None
    bp_wind_yh_file: Optional[str] = None
    pixwin_yh_file: Optional[str] = None
    Nbins_yh: Optional[str] = None   
    Mmin: Optional[str] = None
    Mmax: Optional[str] = None
    zmin: Optional[str] = None
    zmax: Optional[str] = None

    # Load the data
    def initialize(self):
        Mmin = self.Mmin
        Mmax = self.Mmax
        zmin = self.zmin
        zmax = self.zmax
        self.bpwf_yh = np.load(os.path.join(self.data_directory, self.bp_wind_yh_file))[0]
        self.pw_yh = np.loadtxt(os.path.join(self.data_directory, self.pixwin_yh_file))
        Np_yh = self.Nbins_yh
        
        D_yh = np.loadtxt(os.path.join(self.data_directory, self.yxh_data_file)+f'_z_{zmin:.2f}-{zmax:.2f}_MinMass_{Mmin:.1e}_MaxMass_{Mmax:.1e}.txt')
        cov_yh = np.loadtxt(os.path.join(self.data_directory, self.cov_data_file)+f'_z_{zmin:.2f}-{zmax:.2f}_MinMass_{Mmin:.1e}_MaxMass_{Mmax:.1e}.txt')
        print(cov_yh.shape)

        cov_yh= cov_yh[:Np_yh,:Np_yh]


        self.ell_yh = D_yh[0,:Np_yh]
        self.yh = D_yh[1,:Np_yh]
        self.covmat =  cov_yh
        
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        # print("cov:", np.diag(self.covmat))
        self.cl_joint = self.yh
        self.ell_joint = self.ell_yh
        print("cl_joint", self.cl_joint)
        super().initialize()


    def get_requirements(self):
        return {'Cl_yxg':{}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell_joint
        y_data = self.cl_joint
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov

    def _bin(self, ell_theory, cl_theory, ell_data, ellmax, bpwf, Nellbins, pix_win=None, conv2cl=True,):
        """
        Interpolate the theory dl's, and bin according to the bandpower window function (bpwf)
        """
        #interpolate
        # ellmax=int(np.round(ell_data[len(ell_data)-1]))
        # print("ellmax",ellmax)
        new_ell = np.arange(2, ellmax, 1)
        cl_theory_log = np.log(cl_theory)
        f_int =  interp1d(ell_theory, cl_theory_log, fill_value="extrapolate")
        inter_cl_log = np.asarray(f_int(new_ell))
        inter_cl= np.exp(inter_cl_log)
        if conv2cl==True: #go from dls to cls because the bpwf mutliplies by ell*(ell+1)/2pi
            inter_cl= inter_cl*(2.0*np.pi)/(new_ell)/(new_ell+1.0)

        #multiply by the pixel window function (from healpix for given nside)
        if pix_win is not None:
            inter_cl = inter_cl *(pix_win[2:ellmax])**2
        
        #bin according to the bpwf
        cl_binned = np.zeros(Nellbins)
        for i in range (Nellbins):
            wi = bpwf[i]
            # wi starts from ell=2 according to Alex, email 1-9-22; could add ell=0,1, but would contribute nothing to the sum
            cl_binned[i] = np.sum(wi[2:len(inter_cl)+2]*inter_cl)
        #print("clbinned:", cl_binned)
        return ell_data, cl_binned


    def _cl2dl(self, l):
        return l*(l+1)/2/np.pi

    def _get_theory(self, **params_values_dict):
        bpwf_yh = self.bpwf_yh[:,0,:]
        pixwin_yh = self.pw_yh
        Np_yh = self.Nbins_yh
        ellmax_bin_yh = 5200


        yh_all= [] 
        theory_yh = self.provider.get_Cl_yxg()
        ell_theory_yh, cl_1h_theory_yh, cl_2h_theory_yh = theory_yh['ell'], theory_yh['1h'], theory_yh['2h']
        print("ell_theory_yh",ell_theory_yh[:10] )
        print("cl_1h_theory_yh", cl_1h_theory_yh[:10] )
        print("cl_2h_theory_yh", cl_2h_theory_yh[:10] )

        ell_yh_bin, dl_yh_bin_1h = self._bin(ell_theory_yh, np.asarray(cl_1h_theory_yh), self.ell_yh, ellmax_bin_yh, bpwf_yh, Nellbins=Np_yh, pix_win =pixwin_yh,  conv2cl=True)
        ell_yh_bin, dl_yh_bin_2h = self._bin(ell_theory_yh, np.asarray(cl_2h_theory_yh), self.ell_yh, ellmax_bin_yh, bpwf_yh, Nellbins=Np_yh, pix_win =pixwin_yh, conv2cl=True)
        # print("dl_yh_bin:", dl_yh_bin_1h+dl_yh_bin_2h)


        cl_joint = 1e-6*dl_yh_bin_1h+1e-6*dl_yh_bin_2h
        print("cl joint:", cl_joint)

        if np.isnan(cl_joint).any()==True:
            print("Nans in the theory prediction!")
            exit()
        return cl_joint
