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

class GXG_KXG_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    gxg_data_file: Optional[str] = None
    gxk_data_file: Optional[str] = None
    cov_data_file: Optional[str] = None
    bp_wind_gg_file: Optional[str] = None
    bp_wind_gk_file: Optional[str] = None
    pixwind_4096_file: Optional[str] = None
    pixwind_1024_file: Optional[str] = None
    Nbins_kg: Optional[str] = None
    Nbins_gg: Optional[str] = None
    Mbin: Optional[str] = None
    params = {"A_shot_noise": 0}

    # Load the data
    def initialize(self):
        Mbin = self.Mbin
        print("Maglim bin = ", Mbin)

        self.bpwf_kg = np.load(os.path.join(self.data_directory, self.bp_wind_gk_file))[0]
        self.bpwf_gg = np.load(os.path.join(self.data_directory, self.bp_wind_gg_file))[0]
        self.pw_bin_gg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_4096_file))
        self.pw_bin_kg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_1024_file))
        Np_kg = self.Nbins_kg
        Np_gg = self.Nbins_gg
        
        D_kg = np.loadtxt(os.path.join(self.data_directory, self.gxk_data_file))
        D_gg = np.loadtxt(os.path.join(self.data_directory, self.gxg_data_file))
        cov_kg = np.load(os.path.join(self.data_directory, self.cov_data_file)+f'{Mbin}_5_{Mbin}_{Mbin}_dl.pk', allow_pickle=True)[f'bin_{Mbin}_5_{Mbin}_{Mbin}']
        cov_gg = np.load(os.path.join(self.data_directory, self.cov_data_file)+f'{Mbin}_{Mbin}_{Mbin}_{Mbin}_dl.pk', allow_pickle=True)[f'bin_{Mbin}_{Mbin}_{Mbin}_{Mbin}']
        cov_cross = np.load(os.path.join(self.data_directory, self.cov_data_file)+f'{Mbin}_5_{Mbin}_5_dl.pk', allow_pickle=True)[f'bin_{Mbin}_5_{Mbin}_5']

        cov_kg= cov_kg[:Np_kg,:Np_kg]
        cov_gg= cov_gg[:Np_gg,:Np_gg]
        cov_cross= cov_cross[:Np_kg,:Np_gg]

        self.ell_kg = D_kg[0,:Np_kg]
        self.ell_kg_full = D_kg[0,:Np_kg]
        self.kg = D_kg[1,:Np_kg]


        self.ell_gg = D_gg[0,:Np_gg]
        self.ell_gg_full = D_gg[0,:Np_gg]
        self.gg = D_gg[1,:Np_gg]

        # print("ell ola kg :", self.ell_kg)
        # print("kg ola:", self.kg)
        # print("kg shape: ", self.kg.shape)
        Npoints = Np_gg + Np_kg

        self.covmat =  np.block([[cov_kg, cov_cross],[cov_cross, cov_gg]])
        print(self.covmat.shape)
        
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        # print("cov:", np.diag(self.covmat))
        ###Combine into 1 data vector
        self.cl_joint = np.concatenate((self.kg, self.gg), axis=0)
        self.ell_joint = np.concatenate((self.ell_kg, self.ell_gg), axis=0)
        super().initialize()


    def get_requirements(self):
        return {'Cl_gxg':{},'Cl_kgxg':{}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell_joint
        y_data = self.cl_joint
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov

    def _bin(self, ell_theory, cl_theory, ell_data, ellmax, bpwf, pix_win, Nellbins, conv2cl=True,):
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
        inter_cl = inter_cl*(pix_win[2:ellmax])**2
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
        bpwf_gg = self.bpwf_gg[:,0,:]
        pixwin_gg = self.pw_bin_gg
        pixwin_kg = self.pw_bin_kg
        Np_kg = self.Nbins_kg
        Np_gg = self.Nbins_gg
        ellmax_bin_gg = 2200
        ellmax_bin_kg = 2200
        A_shotnoise = params_values_dict['A_shot_noise']
        shot_noise = A_shotnoise*1.e-7

        kg_all, gg_all= [], [],
  

        theory_kg = self.provider.get_Cl_kgxg()
        theory_gg = self.provider.get_Cl_gxg()
        ell_theory_kg, cl_1h_theory_kg, cl_2h_theory_kg = theory_kg['ell'], theory_kg['1h'], theory_kg['2h']
        ell_theory_gg, cl_1h_theory_gg, cl_2h_theory_gg = theory_gg['ell'], theory_gg['1h'], theory_gg['2h']

        ell_kg_bin, dl_kg_bin_1h = self._bin(ell_theory_kg, np.asarray(cl_1h_theory_kg), self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg,  Nellbins=Np_kg, conv2cl=True)
        ell_kg_bin, dl_kg_bin_2h = self._bin(ell_theory_kg, np.asarray(cl_2h_theory_kg), self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
        ell_gg_bin, dl_gg_bin_1h = self._bin(ell_theory_gg, np.asarray(cl_1h_theory_gg), self.ell_gg_full, ellmax_bin_gg, bpwf_gg, pixwin_gg, Nellbins=Np_gg, conv2cl=True)
        ell_gg_bin, dl_gg_bin_2h = self._bin(ell_theory_gg, np.asarray(cl_2h_theory_gg), self.ell_gg_full, ellmax_bin_gg, bpwf_gg, pixwin_gg,  Nellbins=Np_gg, conv2cl=True)
        # print("dl_kg_bin:", dl_kg_bin_1h+dl_kg_bin_2h)
        # print("dl_gg_bin:", dl_gg_bin_1h+dl_gg_bin_2h)

        kg_all.append(dl_kg_bin_1h+dl_kg_bin_2h)
        gg_all.append(dl_gg_bin_1h+dl_gg_bin_2h+shot_noise* self._cl2dl(ell_gg_bin))
        # print("gg: ", gg_all)
        # print("kg: ", kg_all)

        cl_joint = np.concatenate((np.concatenate(kg_all), np.concatenate(gg_all)), axis=0)
        # print("cl joint:", cl_joint)

        if np.isnan(cl_joint).any()==True:
            print("Nans in the theory prediction!")
            exit()
        return cl_joint
