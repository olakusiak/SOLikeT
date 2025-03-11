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

class KXG_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    gxk_data_file: Optional[str] = None
    cov_kg_data_file: Optional[str] = None
    bp_wind_gk_file: Optional[str] = None
    Nbins_kg: Optional[str] = None

    # Load the data
    def initialize(self):
        # self.s = np.loadtxt(os.path.join(self.data_directory, self.s_file))
        self.bpwf_kg = np.load(os.path.join(self.data_directory, self.bp_wind_gk_file))[0]

        Np_kg = self.Nbins_kg
        D_kg = np.loadtxt(os.path.join(self.data_directory, self.gxk_data_file))
        cov_kg = np.loadtxt(os.path.join(self.data_directory, self.cov_kg_data_file))
        cov_kg= cov_kg[:Np_kg,:Np_kg]


        self.ell_kg = D_kg[0,:Np_kg]
        self.ell_kg_full = D_kg[0,:Np_kg]
        self.kg = D_kg[1,:Np_kg]

        print("ell ola kg :", self.ell_kg)
        print("kg ola:", self.kg)
        print("kg shape: ", self.kg.shape)
        Npoints =  Np_kg

        self.covmat =  cov_kg
        print(self.covmat.shape)
        
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        # print("cov:", np.diag(self.covmat))
        ###Combine into 1 data vector
        self.cl_joint =  self.kg
        self.ell_joint = self.ell_kg
        super().initialize()


    def get_requirements(self):
        return {'Cl_kgxg':{}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell_joint
        y_data = self.cl_joint
        print("data")
        print(x_data, y_data)
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov

    def _bin(self, ell_theory, cl_theory, ell_data, ellmax, bpwf,  Nellbins, conv2cl=True,):
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
        bpwf_kg = self.bpwf_kg[:,0,:]
        Np_kg = self.Nbins_kg
        ellmax_bin_kg = 2200

        kg_all,kg_1h_all, kg_2h_all = [], [], [],

        theory_kg = self.provider.get_Cl_kgxg()
        ell_theory_kg, cl_1h_theory_kg, cl_2h_theory_kg = theory_kg['ell'], theory_kg['1h'], theory_kg['2h']
        print("cl_1h_theory_kg:", cl_1h_theory_kg)
        print("cl_2h_theory_kg:", cl_2h_theory_kg)
        ell_kg_bin, dl_kg_bin_1h = self._bin(ell_theory_kg, np.asarray(cl_1h_theory_kg), self.ell_kg_full, ellmax_bin_kg, bpwf_kg,  Nellbins=Np_kg, conv2cl=True)
        ell_kg_bin, dl_kg_bin_2h = self._bin(ell_theory_kg, np.asarray(cl_2h_theory_kg), self.ell_kg_full, ellmax_bin_kg, bpwf_kg,  Nellbins=Np_kg, conv2cl=True)
        print("dl_kg_bin_1h:", dl_kg_bin_1h)
        print("dl_kg_bin_2h:", dl_kg_bin_2h)

        cl_joint = dl_kg_bin_1h+dl_kg_bin_2h
        print("cl joint:", cl_joint)

        if np.isnan(cl_joint).any()==True:
            print("Nans in the theory prediction!")
            exit()
        return cl_joint
