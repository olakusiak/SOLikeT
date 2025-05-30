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

class YXY_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    yxy_data_file: Optional[str] = None
    cov_data_file: Optional[str] = None
    bp_wind_yy_file: Optional[str] = None
    Nbins_yy: Optional[str] = None
    Mbin: Optional[str] = None

    # Load the data
    def initialize(self):

        self.bpwf_yy = np.load(os.path.join(self.data_directory, self.bp_wind_yy_file))[0]
        Np_yy = self.Nbins_yy
        

        D_yy = np.loadtxt(os.path.join(self.data_directory, self.yxy_data_file))
        cov_yy = np.loadtxt(os.path.join(self.data_directory, self.cov_data_file) )

        cov_yy= cov_yy[:Np_yy,:Np_yy]

        self.ell_yy = D_yy[0,:Np_yy]
        self.ell_yy_full = D_yy[0,:Np_yy]
        self.yy = D_yy[1,:Np_yy]

        # print("ell ola gg :", self.ell_gg)
        # print("gg ola:", self.gg)
        # print("gg shape: ", self.gg.shape)
        Npoints = Np_yy 

        self.covmat =   cov_yy
        print(self.covmat.shape)
        
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        # print("cov:", np.diag(self.covmat))
        ###Combine into 1 data vector
        self.cl_joint = self.yy
        self.ell_joint = self.ell_yy
        super().initialize()


    def get_requirements(self):
        return {'Cl_sz':{}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell_joint
        y_data = self.cl_joint
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov

    def _bin(self, ell_theory, cl_theory, ell_data, ellmax, bpwf, Nellbins, conv2cl=True,):
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
        inter_cl = inter_cl
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

        bpwf_yy = self.bpwf_yy[:,0,:]

        Np_yy = self.Nbins_yy
        ellmax_bin_yy = 5200


        yy_all= []
  


        theory_yy = self.provider.get_Cl_sz()
        ell_theory_yy, cl_1h_theory_yy, cl_2h_theory_yy = theory_yy['ell'], theory_yy['1h'], theory_yy['2h']

        ell_yy_bin, dl_yy_bin_1h = self._bin(ell_theory_yy, np.asarray(cl_1h_theory_yy), self.ell_yy_full, ellmax_bin_yy, bpwf_yy, Nellbins=Np_yy, conv2cl=True)
        ell_yy_bin, dl_yy_bin_2h = self._bin(ell_theory_yy, np.asarray(cl_2h_theory_yy), self.ell_yy_full, ellmax_bin_yy, bpwf_yy, Nellbins=Np_yy, conv2cl=True)
        

        cl_joint =  1e-12* (dl_yy_bin_1h+dl_yy_bin_2h)
        # print("cl joint:", cl_joint)

        if np.isnan(cl_joint).any()==True:
            print("Nans in the theory prediction!")
            exit()
        return cl_joint
