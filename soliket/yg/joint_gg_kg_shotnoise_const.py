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


class joint_KXG_GXG_PS_constSN_Likelihood(GaussianLikelihood):
    sz_data_directory: Optional[str] = None
    kxg_ps_file: Optional[str] = None
    gxg_ps_file: Optional[str] = None
    cov_joint_ps_file_kg_kg: Optional[str] = None
    cov_joint_ps_file_kg_gg: Optional[str] = None
    cov_joint_ps_file_gg_gg: Optional[str] = None
    s_file: Optional[str] = None
    params = {"A_shot_noise": 0}


    # Load the templates
    def initialize(self):
        self.data_directory = self.sz_data_directory
        self.datafile_gg = self.gxg_ps_file
        self.datafile_kg = self.kxg_ps_file
        self.covfile_kg_kg = self.cov_joint_ps_file_kg_kg
        self.covfile_kg_gg = self.cov_joint_ps_file_kg_gg
        self.covfile_gg_gg = self.cov_joint_ps_file_gg_gg
        self.s = np.loadtxt(os.path.join(self.data_directory, self.s_file))

        D = np.loadtxt(os.path.join(self.data_directory, self.datafile_gg))
        D_kg = np.loadtxt(os.path.join(self.data_directory, self.datafile_kg))
        cov_kg_kg = np.loadtxt(os.path.join(self.data_directory, self.covfile_kg_kg))
        cov_kg_gg = np.loadtxt(os.path.join(self.data_directory, self.covfile_kg_gg))
        cov_gg_gg = np.loadtxt(os.path.join(self.data_directory, self.covfile_gg_gg))
        cov_kg_kg = cov_kg_kg[:10,:10]
        cov_gg_gg = cov_gg_gg[1:10,1:10] #cut the first data point
        cov_kg_gg = cov_kg_gg[:10,1:10]
        #multipoles of bin centre
        self.ell_full = D_kg[0,:]
        self.ell = np.concatenate((D[0,1:10], D_kg[0,:10]), axis=0)
        #self.ell = self.ell1 = D[:10,0]
        #print("ell alex: ", self.ell)
        #print("Data alex kg : ", self.ell_kg)
        # gxg + foregrounds
        self.cl_joint = np.concatenate((D[1,1:10], D_kg[1,:10]), axis=0)
        #self.cl_joint = (D[:10,1])
        #print("cl alex: ", self.cl_joint)

        #self.sigma_tot = D[:10,2]
        #self.sigma_tot_kg = D_kg[2,:10]
        #cov_zeros=np.zeros(cov_kg_gg.shape)
        A=np.concatenate((cov_gg_gg , cov_kg_gg), axis=0)
        B=np.concatenate((cov_kg_gg.T, cov_kg_kg, ), axis=0)
        C=np.concatenate((A,B), axis=1)

        #self.cvg = np.asarray(self.sigma_tot)**2.
        #self.cvg = np.diag(self.cvg)
        self.cvg = C
        #print(C.shape)
        self.covmat = self.cvg
        # now compute the iverse and det of covariance matrix
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        #print('[reading ref gxg data files] read-in completed.')
        super().initialize()


    def get_requirements(self):
        return {"Cl_gxg": {}, "Cl_gxmu": {}, "Cl_muxmu": {}, "Cl_kxmu": {}, "Cl_kxg": {}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell
        y_data = self.cl_joint
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov

    def binning(self, ell_class, dl_class, ell_alex, Nellbins=9, ellmin = 100.5, ellmax = 1000.5,):
        #interpolate and transform to cl's (Alex data is in cl's)
        # in log
        dl_class = np.log(dl_class)
        f_kg = interp1d(ell_class, dl_class)
        new_ell = np.arange(2, ell_alex[10], 1) # up to 1051.5
        #print("new_ell:", new_ell)
        inter_dls=np.asarray(f_kg(new_ell))
        inter_dls = np.exp(inter_dls)
        inter_cls = inter_dls*(2.0*np.pi)/(new_ell)/(new_ell+1.0)
        #binning
        binbounds = np.linspace(ellmin, ellmax, num=Nellbins+1, endpoint=True, dtype=int)
        binbounds = np.append([18], binbounds) #random but works
        ellsubarrs = np.split(new_ell, binbounds)
        clsubarrs = np.split(inter_cls, binbounds)
        ellbinned = np.zeros(len(binbounds)+1)
        clbinned = np.zeros(len(binbounds)+1)
        for j in range(len(binbounds)+1):
            ellbinned[j] = np.mean(ellsubarrs[j])
            clbinned[j] = np.mean(clsubarrs[j])
        return ellbinned[1:-1], clbinned[1:-1]

    def _get_theory(self, **params_values):
        # s_blue =  0.455
        # s_green =  0.648
        # s_red = 0.842
        A_shot_noise =  3.8836691E-01
        s=self.s
        shot_noise  = A_shot_noise * 1e-7
        #print("shot_noise: ", shot_noise )
        ########
        # Cl_gxg
        ########
        theory = self.theory.get_Cl_gxg()
        cl_ell_theory = theory['ell']
        dl_1h_theory = theory['1h']
        dl_2h_theory = theory['2h']
        ell = np.asarray(list(cl_ell_theory))
        dl_gg_theory = np.asarray(list(dl_1h_theory)) + np.asarray(list(dl_2h_theory))
        ell_gg_binned, cl_gg_binned = self.binning(ell, dl_gg_theory, self.ell_full, Nellbins=9, ellmin = 100.5, ellmax = 1000.5)
        ell_gg_binned, cl_gg_binned = ell_gg_binned[1:], cl_gg_binned[1:]
        #print("cl_gg: ", cl_gg_binned)
        #print("ell_gg: ", ell_gg_binned)
        ########
        # Cl_gxmu
        ########
        theory_gm = self.theory.get_Cl_gxmu()
        cl_ell_theory_gm = theory_gm['ell']

        dl_1h_theory_gm = theory_gm['1h']
        dl_2h_theory_gm = theory_gm['2h']
        ell = np.asarray(list(cl_ell_theory))
        dl_gm_theory = np.asarray(list(dl_1h_theory_gm)) + np.asarray(list(dl_2h_theory_gm))
        #print("dl_gg_theory: ", dl_gg_theory)
        ell_gm_binned, cl_gm_binned = self.binning(ell, dl_gm_theory, self.ell_full, Nellbins=9, ellmin = 100.5, ellmax = 1000.5)
        ell_gm_binned, cl_gm_binned = ell_gm_binned[1:], cl_gm_binned[1:]

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
        ell_mm_binned, cl_mm_binned = self.binning(ell, dl_mm_theory, self.ell_full, Nellbins=9, ellmin = 100.5, ellmax = 1000.5)
        ell_mm_binned, cl_mm_binned = ell_mm_binned[1:], cl_mm_binned[1:]
        #print('cl mm: ', cl_mm_binned)
        #print("cl_gg: ", cl_gg_binned + 2*(5*s-2)*cl_gm_binned + (5*s-2)*(5*s-2)*cl_mm_binned)

        ########
        # Cl_kxg
        ########
        theory_kg = self.theory.get_Cl_kxg()
        ell_theory_kg = theory_kg['ell']
        dl_1h_theory_kg = theory_kg['1h']
        #print(dl_1h_theory)
        dl_2h_theory_kg = theory_kg['2h']
        dl_kg_theory = np.asarray(list(dl_1h_theory_kg)) + np.asarray(list(dl_2h_theory_kg))
        ell_theory_kg = np.asarray(list(ell_theory_kg))
        ell_kg_binned, cl_kg_binned = self.binning(ell_theory_kg, dl_kg_theory, self.ell_full , Nellbins=9, ellmin = 100.5, ellmax = 1000.5)
        #print("ell_kg: ", ell_kg_binned)
        ##########
        # Cl_kxmu
        ##########
        theory_km = self.theory.get_Cl_kxmu()
        cl_ell_theory_km = theory_km['ell']
        dl_1h_theory_km = theory_km['1h']
        dl_2h_theory_km = theory_km['2h']
        dl_km_theory = np.asarray(list(dl_1h_theory_km)) + np.asarray(list(dl_2h_theory_km))
        ell_theory = np.asarray(list(cl_ell_theory_km))
        ell_km_binned, cl_km_binned = self.binning(ell_theory, dl_km_theory, self.ell_full , Nellbins=9, ellmin = 100.5, ellmax = 1000.5)
        #print('cl kg: ', cl_kg_binned + (5*s-2)*cl_km_binned)

        kg = cl_kg_binned + (5*s-2)*cl_km_binned
        gg = cl_gg_binned + 2*(5*s-2)*cl_gm_binned + (5*s-2)*(5*s-2)*cl_mm_binned + shot_noise
        cl_joint = np.concatenate((gg, kg), axis=0)
        #print("joint: ", cl_joint)
        return cl_joint
