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

class YXG_GXG_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    yxg_data_file: Optional[str] = None
    gxg_data_file: Optional[str] = None
    cov_data_file: Optional[str] = None
    bp_wind_yg_file: Optional[str] = None
    bp_wind_gg_file: Optional[str] = None
    pixwind_4096_file: Optional[str] = None
    Nbins_gg: Optional[str] = None
    Nbins_yg: Optional[str] = None
    Mbin: Optional[str] = None

    # Load the data
    def initialize(self):
        Mbin = self.Mbin
        print("Maglim bin = ", Mbin)

        self.bpwf_gg = np.load(os.path.join(self.data_directory, self.bp_wind_gg_file))[0]
        self.bpwf_yg = np.load(os.path.join(self.data_directory, self.bp_wind_yg_file))[0]
        self.pw_bin_yg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_4096_file))
        self.pw_bin_gg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_4096_file))
        Np_gg = self.Nbins_gg
        Np_yg = self.Nbins_yg
        
        D_gg = np.loadtxt(os.path.join(self.data_directory, self.gxg_data_file))
        D_yg = np.loadtxt(os.path.join(self.data_directory, self.yxg_data_file))
        cov_gg = np.load(os.path.join(self.data_directory, self.cov_data_file)+f'{Mbin}_5_{Mbin}_{Mbin}_dl.pk', allow_pickle=True)[f'bin_{Mbin}_5_{Mbin}_{Mbin}']
        cov_yg = np.load(os.path.join(self.data_directory, self.cov_data_file)+f'{Mbin}_{Mbin}_{Mbin}_{Mbin}_dl.pk', allow_pickle=True)[f'bin_{Mbin}_{Mbin}_{Mbin}_{Mbin}']
        cov_cross = np.load(os.path.join(self.data_directory, self.cov_data_file)+f'{Mbin}_5_{Mbin}_5_dl.pk', allow_pickle=True)[f'bin_{Mbin}_5_{Mbin}_5']

        cov_gg= cov_gg[:Np_gg,:Np_gg]
        cov_yg= cov_yg[:Np_yg,:Np_yg]
        cov_cross= cov_cross[:Np_gg,:Np_yg]

        self.ell_gg = D_gg[0,:Np_gg]
        self.ell_gg_full = D_gg[0,:Np_gg]
        self.gg = D_gg[1,:Np_gg]


        self.ell_yg = D_yg[0,:Np_yg]
        self.ell_yg_full = D_yg[0,:Np_yg]
        self.yg = D_yg[1,:Np_yg]

        # print("ell ola gg :", self.ell_gg)
        # print("gg ola:", self.gg)
        # print("gg shape: ", self.gg.shape)
        Npoints = Np_yg + Np_gg

        self.covmat =  np.block([[cov_gg, cov_cross],[cov_cross, cov_yg]])
        print(self.covmat.shape)
        
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        # print("cov:", np.diag(self.covmat))
        ###Combine into 1 data vector
        self.cl_joint = np.concatenate((self.gg, self.yg), axis=0)
        self.ell_joint = np.concatenate((self.ell_gg, self.ell_yg), axis=0)
        super().initialize()


    def get_requirements(self):
        return {'Cl_yxg':{},'Cl_gxg':{}}

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
        bpwf_gg = self.bpwf_gg[:,0,:]
        bpwf_yg = self.bpwf_yg[:,0,:]
        pixwin_yg = self.pw_bin_yg
        pixwin_gg = self.pw_bin_gg
        Np_gg = self.Nbins_gg
        Np_yg = self.Nbins_yg
        ellmax_bin_yg = 2200
        ellmax_bin_gg = 2200

        gg_all, yg_all= [], [],
  

        theory_gg = self.provider.get_Cl_ggxg()
        theory_yg = self.provider.get_Cl_yxg()
        ell_theory_gg, cl_1h_theory_gg, cl_2h_theory_gg = theory_gg['ell'], theory_gg['1h'], theory_gg['2h']
        ell_theory_yg, cl_1h_theory_yg, cl_2h_theory_yg = theory_yg['ell'], theory_yg['1h'], theory_yg['2h']

        ell_gg_bin, dl_gg_bin_1h = self._bin(ell_theory_gg, np.asarray(cl_1h_theory_gg), self.ell_gg_full, ellmax_bin_gg, bpwf_gg, pixwin_gg,  Nellbins=Np_gg, conv2cl=True)
        ell_gg_bin, dl_gg_bin_2h = self._bin(ell_theory_gg, np.asarray(cl_2h_theory_gg), self.ell_gg_full, ellmax_bin_gg, bpwf_gg, pixwin_gg, Nellbins=Np_gg, conv2cl=True)
        ell_yg_bin, dl_yg_bin_1h = self._bin(ell_theory_yg, np.asarray(cl_1h_theory_yg), self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg, Nellbins=Np_yg, conv2cl=True)
        ell_yg_bin, dl_yg_bin_2h = self._bin(ell_theory_yg, np.asarray(cl_2h_theory_yg), self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg,  Nellbins=Np_yg, conv2cl=True)
        print("dl_gg_bin:", dl_gg_bin_1h+dl_gg_bin_2h)
        print("dl_yg_bin:", dl_yg_bin_1h+dl_yg_bin_2h)

        gg_all.append(dl_gg_bin_1h+dl_gg_bin_2h)
        yg_all.append(dl_yg_bin_1h+dl_yg_bin_2h)
        # print("yg: ", yg_all)
        # print("gg: ", gg_all)

        cl_joint = np.concatenate((np.concatenate(gg_all), np.concatenate(yg_all)), axis=0)
        # print("cl joint:", cl_joint)

        if np.isnan(cl_joint).any()==True:
            print("Nans in the theory prediction!")
            exit()
        return cl_joint
