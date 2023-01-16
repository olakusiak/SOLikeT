"""
.. module:: gxg

:Synopsis: Definition of simplistic y-map power spectrum likelihood using pre-computed frequency dependent  templates
           for gxg powerspectrum (A_gxg), and foregrounds, e.g., A_cib, A_ir, A_rs, A_cn.

:running boris: $ /usr/local/anaconda3/bin/mpirun -np 4 /usr/local/anaconda3/bin/cobaya-run soliket/ymap/input_files/gxg_ps.yaml -f
:running ola: $ /Users/boris/opt/anaconda3/bin/mpirun -np 4 /Users/boris/opt/anaconda3/bin/cobaya-run soliket/ymap/input_files/gxg_ps_template.yaml -f

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


class GXG_GXM_MXM_PS_Likelihood(GaussianLikelihood):
    sz_data_directory: Optional[str] = None
    gxg_ps_file: Optional[str] = None
    cov_joint_ps_file_gg_gg: Optional[str]  = None
    s_file: Optional[str]  =  None
    params = {"A_shot_noise": 0}
    wind_funct_file_gg: Optional[str] = None
    tranfer_funct_file_gg: Optional[str] = None

    # Load the templates
    def initialize(self):

        self.data_directory = self.sz_data_directory
        self.datafile = self.gxg_ps_file
        self.covfile = self.cov_joint_ps_file_gg_gg
        #print(self.data_directory)
        #print(self.datafile)
        cov = np.loadtxt(os.path.join(self.data_directory, self.covfile))
        D = np.loadtxt(os.path.join(self.data_directory, self.datafile))
        cov=cov[1:10,1:10]
        self.ell = D[0,1:10]
        self.ell_full = D[0,]
        self.gxgAndFg = D[1,1:10]

        # s for the lensing magnification
        self.s = np.loadtxt(os.path.join(self.data_directory, self.s_file))
        #bandpower window functions
        self.wind_gg = np.loadtxt(os.path.join(self.data_directory, self.wind_funct_file_gg))
        #transfer functions
        self.trans_gg = np.loadtxt(os.path.join(self.data_directory, self.tranfer_funct_file_gg))
        #print("trans", self.trans_gg)
        #print("wind", self.wind_gg)
        #print(self.s)
        #print("ell alex: ", self.ell)
        #print("cl alex: ", self.gxgAndFg)
        #self.sigma_tot = D[:,2]
        #print("sigma^2 : ", self.sigma_tot**2)
        #self.cvg = np.asarray(self.sigma_tot)**2.
        #self.cvg = np.diag(self.cvg)
        self.cvg = cov
        self.covmat = self.cvg

        # now compute the iverse and det of covariance matrix
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print("eig:", np.linalg.eig(self.covmat))
        #print('[reading ref gxg data files] read-in completed.')
        super().initialize()


    def get_requirements(self):
        return {"Cl_gxg": {}, "Cl_gxmu": {}, "Cl_muxmu": {}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell
        y_data = self.gxgAndFg
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov
    def binning(self, ell_class, dl_class, ell_alex, wind_func, Nellbins=9, ellmin = 100.5, ellmax = 1000.5,):
        #interpolate and transform to cl's (Alex data is in cl's)
        dl_class = np.log(dl_class)  #in log
        f_kg = interp1d(ell_class, dl_class)
        new_ell = np.arange(2, ell_alex[15], 1) # up to 1051.5
        #print(new_ell)
        inter_dls=np.asarray(f_kg(new_ell))
        inter_dls = np.exp(inter_dls)
        inter_cls = inter_dls*(2.0*np.pi)/(new_ell)/(new_ell+1.0)
        #binning
        clbinned = np.zeros(10)
        for i in range (10):
            #bandpower window function / C_i_binned = \sum_{ell} W_i(ell) C_ell
            wi = wind_func[i]
            # wi starts from ell=2 according to Alex, email 1-9-22; could add ell=0,1, but would contribute nothing to the sum
            ci_binned = np.sum(wi[2:1504]*inter_cls[:1502])
            #ci_binned = np.sum(wi[2:1504]*inter_cls[:1502])
            #ci_binned = np.sum(wi[2:5852]*inter_cls[:6002]) #cutting 1500 only changes chi2 by 0.04
            clbinned[i]=ci_binned
            #clbinned.append(ci_binned)
        #print(clbinned)
        return ell_alex[:10], clbinned[:10]

    # def binning_old(self, ell_class, dl_class, ell_alex, Nellbins=9, ellmin = 100.5, ellmax = 1000.5,):
    #     #interpolate and transform to cl's (Alex data is in cl's)
    #     # in log
    #     dl_class = np.log(dl_class)
    #     f_kg = interp1d(ell_class, dl_class)
    #     new_ell = np.arange(2, ell_alex[10], 1) # up to 1051.5
    #     inter_dls=np.asarray(f_kg(new_ell))
    #     inter_dls = np.exp(inter_dls)
    #     inter_cls = inter_dls*(2.0*np.pi)/(new_ell)/(new_ell+1.0)
    #
    #     #binning
    #     binbounds = np.linspace(ellmin, ellmax, num=Nellbins+1, endpoint=True, dtype=int)
    #     binbounds = np.append([18], binbounds) #random but works
    #     ellsubarrs = np.split(new_ell, binbounds)
    #     clsubarrs = np.split(inter_cls, binbounds)
    #     ellbinned = np.zeros(len(binbounds)+1)
    #     clbinned = np.zeros(len(binbounds)+1)
    #     for j in range(len(binbounds)+1):
    #         ellbinned[j] = np.mean(ellsubarrs[j])
    #         clbinned[j] = np.mean(clsubarrs[j])
    #     return ellbinned[2:-1], clbinned[2:-1]

    def _get_theory(self, **params_values):
        A_shot_noise = params_values['A_shot_noise']
        s=self.s
        shot_noise  = A_shot_noise * 1e-7
        trans_gg = self.trans_gg
        wind_gg = self.wind_gg
        # print("s ", s )
        #print("shot_noise: ", A_shot_noise )
        #print("trans:", trans_gg)
        # #print("wind:", wind_gg)
        ########
        # Cl_gxg
        ########
        theory = self.theory.get_Cl_gxg()
        #print(theory)
        cl_ell_theory = theory['ell']
        dl_1h_theory = theory['1h']
        dl_2h_theory = theory['2h']
        #print(dl_1h_theory)
        ell = np.asarray(list(cl_ell_theory))
        dl_gg_theory = np.asarray(list(dl_1h_theory)) + np.asarray(list(dl_2h_theory))

        ell_gg_binned, cl_gg_binned = self.binning(ell, dl_gg_theory, self.ell_full, wind_gg, Nellbins=9, ellmin = 100.5, ellmax = 1000.5)
        # print(cl_ell_theory)
        # print('cl gg theory: ', dl_1h_theory)
        #print('cl gg: ', cl_gg_binned)
        # print('ell gg: ', ell_gg_binned)
        # ########
        # Cl_gxmu
        ########
        theory_gm = self.theory.get_Cl_gxmu()
        cl_ell_theory_gm = theory_gm['ell']
        dl_1h_theory_gm = theory_gm['1h']
        dl_2h_theory_gm = theory_gm['2h']
        ell = np.asarray(list(cl_ell_theory))
        dl_gm_theory = np.asarray(list(dl_1h_theory_gm)) + np.asarray(list(dl_2h_theory_gm))
        ell_gm_binned, cl_gm_binned = self.binning(ell, dl_gm_theory, self.ell_full, wind_gg, Nellbins=9, ellmin = 100.5, ellmax = 1000.5)
        #print('cl gm: ', cl_gm_binned)
        ########
        # Cl_muxmu
        ########
        theory_mm = self.theory.get_Cl_muxmu()
        cl_ell_theory_mm = theory_mm['ell']
        dl_1h_theory_mm = theory_mm['1h']
        dl_2h_theory_mm = theory_mm['2h']
        dl_mm_theory = np.asarray(list(dl_1h_theory_mm)) + np.asarray(list(dl_2h_theory_mm))
        ell_theory = np.asarray(list(cl_ell_theory_mm))
        ell_mm_binned, cl_mm_binned = self.binning(ell, dl_mm_theory, self.ell_full, wind_gg, Nellbins=9, ellmin = 100.5, ellmax = 1000.5)
        #print('cl mm: ', cl_mm_binned)
        #print("total cl + SN: ", cl_gg_binned + 2*(5*s-2)*cl_gm_binned + (5*s-2)*(5*s-2)*cl_mm_binned + shot_noise)
        #print("total ell: ", ell_mm_binned)
        cl_gg = cl_gg_binned + 2*(5*s-2)*cl_gm_binned + (5*s-2)*(5*s-2)*cl_mm_binned + shot_noise
        #print(cl_gg)
        #apply TF
        gg = cl_gg*trans_gg
        #cut the first point in gg
        gg = gg[1:]
        #print(gg)
        print(gg)
        return gg
