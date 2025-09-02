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
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.special import jv
# from scipy.integrate import simps

class YXG_GXG_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    Mbin: Optional[str] = None
    bp_wind_yg_file: Optional[str] = None
    bp_wind_gg_file: Optional[str] = None
    Nbins_yg: Optional[str] = None
    Nbins_gg: Optional[str] = None
    # Load the data
    def initialize(self):
        Mbin = int(self.Mbin - 1 )
        print("Mbin=", Mbin)
        Np_yg = self.Nbins_yg
        Np_gg = self.Nbins_gg
        Npoints = Np_gg + Np_yg
        nspt = 9
        self.bpwf_yg = np.load(os.path.join(self.data_directory, self.bp_wind_yg_file))[0]
        self.bpwf_gg = np.load(os.path.join(self.data_directory, self.bp_wind_gg_file))[0]
    
        # Load the data
        D_yg = np.load(os.path.join(self.data_directory, f'cls_y_spt_z{Mbin}.npz'))
        D_gg = np.load(os.path.join(self.data_directory, f'cls_z{Mbin}_z{Mbin}.npz'))
        self.ell_yg, self.yg, _ = D_yg['ls'][nspt:nspt+Np_yg], D_yg['cls'][nspt:nspt+Np_yg], D_yg['nls'][nspt:nspt+Np_yg]
        self.ell_gg, self.gg, self.shot_gg = D_gg['ls'][:Np_gg], D_gg['cls'][:Np_gg], D_gg['nls'][:Np_gg]
        print(self.ell_gg)
        print("gg data:", self.gg-self.shot_gg) 
        print("yg data:", self.yg)

        # Covariance matrix
        cov_ygyg = np.load(os.path.join(self.data_directory,f"cov/cov_jk_z{Mbin}_y_spt_z{Mbin}_y_spt.npz"))['cov'][nspt:nspt+Np_yg, nspt:nspt+Np_yg]
        cov_gggg = np.load(os.path.join(self.data_directory,f"cov/cov_jk_z{Mbin}_z{Mbin}_z{Mbin}_z{Mbin}.npz"))['cov'][nspt:nspt+Np_gg, nspt:nspt+Np_gg]
        cov_yggg = np.load(os.path.join(self.data_directory,f"cov/cov_jk_z{Mbin}_z{Mbin}_z{Mbin}_y_spt.npz"))['cov'][nspt:nspt+Np_gg, nspt:nspt+Np_yg]
        self.cov = np.block([
                    [cov_gggg, cov_yggg],
                    [cov_yggg.T, cov_ygyg]  # AB.T if the covariance is symmetric
                    ])
    
        print("cov shape",self.cov.shape)
       
        self.inv_covmat = np.linalg.inv(self.cov)
        self.det_covmat = np.linalg.det(self.cov)

        self.cl_joint = np.concatenate((self.gg-self.shot_gg, self.yg), axis=0)
        self.ell_joint = np.concatenate((self.ell_gg, self.ell_yg), axis=0)
        super().initialize()

    def get_requirements(self):
        return {"Cl_yxg": {}, "Cl_gxg": {}}

    # this is the data to fit
    def _get_data(self): 
        x_data = self.ell_joint
        y_data = self.cl_joint
        return x_data, y_data

    def _get_cov(self):
        cov = self.cov
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
            cl_binned[i] = np.sum(wi*inter_cl)
        #print("clbinned:", cl_binned)
        return ell_data, cl_binned
    

    def _cl2dl(self, l):
        return l*(l+1)/2/np.pi


    def _get_theory(self, **params_values_dict):
        bpwf_gg = self.bpwf_gg[:,0,:]
        bpwf_yg = self.bpwf_yg[:,0,:]

        Np_gg = self.Nbins_gg
        Np_yg = self.Nbins_yg
        ellmax_bin_yg = 6142 +2
        ellmax_bin_gg = 6142 +2

        gg_all, yg_all= [], [],
  

        theory_gg = self.provider.get_Cl_gxg()
        theory_yg = self.provider.get_Cl_yxg()
        ell_theory_gg, cl_1h_theory_gg, cl_2h_theory_gg = theory_gg['ell'], theory_gg['1h'], theory_gg['2h']
        ell_theory_yg, cl_1h_theory_yg, cl_2h_theory_yg = theory_yg['ell'], theory_yg['1h'], theory_yg['2h']

        ell_gg_bin, dl_gg_bin_1h = self._bin(ell_theory_gg, np.asarray(cl_1h_theory_gg), self.ell_gg, ellmax_bin_gg, bpwf_gg,  Nellbins=Np_gg, conv2cl=True)
        ell_gg_bin, dl_gg_bin_2h = self._bin(ell_theory_gg, np.asarray(cl_2h_theory_gg), self.ell_gg, ellmax_bin_gg, bpwf_gg, Nellbins=Np_gg, conv2cl=True)
        ell_yg_bin, dl_yg_bin_1h = self._bin(ell_theory_yg, np.asarray(cl_1h_theory_yg), self.ell_yg, ellmax_bin_yg, bpwf_yg,  Nellbins=Np_yg, conv2cl=True)
        ell_yg_bin, dl_yg_bin_2h = self._bin(ell_theory_yg, np.asarray(cl_2h_theory_yg), self.ell_yg, ellmax_bin_yg, bpwf_yg,   Nellbins=Np_yg, conv2cl=True)


        gg_all.append(dl_gg_bin_1h+dl_gg_bin_2h)
        yg_all.append(1e-6*dl_yg_bin_1h+1e-6*dl_yg_bin_2h)
        print("yg: ", yg_all)
        print("gg: ", gg_all)

        cl_joint = np.concatenate((np.concatenate(gg_all), np.concatenate(yg_all)), axis=0)
        # print("cl joint:", cl_joint)

        if np.isnan(cl_joint).any()==True:
            print("Nans in the theory prediction!")
            exit()
        return cl_joint

