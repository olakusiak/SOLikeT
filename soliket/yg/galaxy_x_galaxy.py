"""
.. module:: gxg

:Synopsis:  gxg powerspectrum (A_gxg)

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


class GXG_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    gxg_data_file: Optional[str] = None
    cov_gxg_data_file: Optional[str]  = None
    s_file: Optional[str]  =  None
    #params = {"A_shot_noise": 0, "logA_shot_noise": 1}
    params = {"A_shot_noise": 0,}
    #print(params)
    wind_funct_file_gg: Optional[str] = None
    tranfer_funct_file_gg: Optional[str] = None
    pixwind_file: Optional[str] = None
    n_degsq_data: Optional[str] = None
    Nbins: Optional[str] = None

    # Load the templates
    def initialize(self):

        self.Npoints = self.Nbins
        print("Npoints", self.Npoints)
        Npoints = self.Npoints
        self.datafile = self.gxg_data_file
        self.covfile = self.cov_gxg_data_file
        ell, pw_bin = np.loadtxt(os.path.join(self.data_directory, self.pixwind_file))
        self.pw_bin  = pw_bin[:Npoints]


        cov = np.loadtxt(os.path.join(self.data_directory, self.covfile))
        D = np.loadtxt(os.path.join(self.data_directory, self.datafile))

        cov=cov[1:Npoints,1:Npoints]
        self.ell = D[0,1:Npoints]
        self.ell_full = D[0,]
        self.gg = D[1,1:Npoints]
        self.sigma = D[2,1:Npoints]

        # s for the lensing magnification
        self.s = np.loadtxt(os.path.join(self.data_directory, self.s_file))
        #bandpower window functions
        self.wind_gg = np.loadtxt(os.path.join(self.data_directory, self.wind_funct_file_gg))
        #transfer functions
        self.trans_gg = np.loadtxt(os.path.join(self.data_directory, self.tranfer_funct_file_gg))
        self.n_degsq = self.n_degsq_data
        self.cvg = cov
        self.covmat = self.cvg
        # self.covmat = np.asarray(self.sigma)**2.
        # self.covmat = np.diag(self.covmat)

        # now compute the iverse and det of covariance matrix
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print("eig:", np.linalg.eig(self.covmat))
        super().initialize()


    def get_requirements(self):
        return {"Cl_gxg": {}, "Cl_gxmu": {}, "Cl_muxmu": {}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell
        y_data = self.gg
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov
    def binning(self, ell_class, dl_class, ell_alex, wind_func, Nellbins=9):
        #interpolate and transform to cl's (Alex data is in cl's)
        dl_class = np.log(dl_class)  #in log
        f_kg = interp1d(ell_class, dl_class)
        new_ell = np.arange(2, ell_alex[Nellbins], 1) # up to 1051.5
        #print(new_ell)
        inter_dls=np.asarray(f_kg(new_ell))
        inter_dls = np.exp(inter_dls)
        inter_cls = inter_dls*(2.0*np.pi)/(new_ell)/(new_ell+1.0)
        #binning
        clbinned = np.zeros(Nellbins)
        for i in range (Nellbins):
            #bandpower window function / C_i_binned = \sum_{ell} W_i(ell) C_ell
            wi = wind_func[i]
            # wi starts from ell=2 according to Alex, email 1-9-22; could add ell=0,1, but would contribute nothing to the sum
            ci_binned = np.sum(wi[2:len(inter_cls)+2]*inter_cls)
            #ci_binned = np.sum(wi[2:1504]*inter_cls[:1502])
            #ci_binned = np.sum(wi[2:5852]*inter_cls[:6002]) #cutting 1500 only changes chi2 by 0.04
            clbinned[i]=ci_binned
            #clbinned.append(ci_binned)
        #print(clbinned)
        return ell_alex, clbinned

    def _get_theory(self, **params_values):
        A_shot_noise = params_values['A_shot_noise']
        s=self.s
        shot_noise  = A_shot_noise
        pixwin = self.pw_bin
        Npoints = self.Npoints
        deg2_to_sr = 3282.8
        ng = (self.n_degsq*deg2_to_sr)
        #logA_shot_noise = params_values['logA_shot_noise']
        # s=self.s
        # shot_noise  = 10**logA_shot_noise
        #print("shot_noise: ", shot_noise)
        #print("pixwin", pixwin)
        trans_gg = self.trans_gg
        wind_gg = self.wind_gg
        #print("shot_noise: ", A_shot_noise )

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

        ell_gg_binned, cl_gg_binned = self.binning(ell, dl_gg_theory, self.ell_full, wind_gg, Nellbins=Npoints)
        # print(cl_ell_theory)
        # print('cl gg theory: ', dl_1h_theory)
        #vprint('cl gg: ', cl_gg_binned)
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
        ell_gm_binned, cl_gm_binned = self.binning(ell, dl_gm_theory, self.ell_full, wind_gg, Nellbins=Npoints)
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
        ell_mm_binned, cl_mm_binned = self.binning(ell, dl_mm_theory, self.ell_full, wind_gg, Nellbins=Npoints)
        #print('cl mm: ', cl_mm_binned)
        #print("total cl + SN: ", cl_gg_binned + 2*(5*s-2)*cl_gm_binned + (5*s-2)*(5*s-2)*cl_mm_binned + shot_noise)
        #print("total ell: ", ell_mm_binned)
        cl_gg = cl_gg_binned + 2*(5*s-2)*cl_gm_binned + (5*s-2)*(5*s-2)*cl_mm_binned + shot_noise*pixwin**(-2) + 1/ng * (1- pixwin**(-2))
        # print(1/ng * (1- pixwin**(-2)))
        # print(cl_gg)
        #apply TF
        trans = np.append(trans_gg, np.ones(Npoints-len(trans_gg)))
        gg = cl_gg *trans
        #cut the first point in gg
        gg = gg[1:]
        #print(gg)
        return gg

    # def _bin(self, ell_theory, cl_theory, ell_data, bpwf, Nellbins=40, conv2cl=True,):
    #     """
    #     Interpolate the theory dl's, and bin according to the bandpower window function (bpwf)
    #     """
    #     #interpolate
    #     new_ell = np.arange(2, 8000, 1)
    #     cl_theory_log = np.log(cl_theory)
    #     f_int =  interp1d(ell_theory, cl_theory_log)
    #     inter_cl_log = np.asarray(f_int(new_ell))
    #     inter_cl= np.exp(inter_cl_log)
    #     if conv2cl==True: #go from dls to cls because the bpwf mutliplies by ell*(ell+1)/2pi
    #         inter_cl= inter_cl*(2.0*np.pi)/(new_ell)/(new_ell+1.0)
    #
    #     #bin according to the bpwf
    #     cl_binned = np.zeros(Nellbins)
    #     for i in range (Nellbins):
    #         wi = bpwf[i]
    #         # wi starts from ell=2 according to Alex, email 1-9-22; could add ell=0,1, but would contribute nothing to the sum
    #         cl_binned[i] = np.sum(wi[2:len(inter_cl)+2]*inter_cl)
    #     #print("clbinned:", cl_binned)
    #     return ell_data, cl_binned
    #
    #
    # def _get_theory(self, **params_values):
    #     A_shot_noise = params_values['A_shot_noise']
    #     s=self.s
    #     shot_noise  = A_shot_noise * 1e-7
    #
    #     bpwf=self.bpwf
    #     print("bpwf", bpwf)
    #     #print("shot_noise: ", A_shot_noise )
    #
    #     ########
    #     # Cl_gxg
    #     ########
    #     theory = self.theory.get_Cl_gxg()
    #     #print(theory)
    #     cl_ell_theory = theory['ell']
    #     dl_1h_theory = theory['1h']
    #     dl_2h_theory = theory['2h']
    #     #print(dl_1h_theory)
    #     ell_theory = np.asarray((cl_ell_theory))
    #     dl_gg_theory = np.asarray((dl_1h_theory)) + np.asarray((dl_2h_theory))
    #     ell_gg_binned, cl_gg_binned = self._bin(ell_theory, dl_gg_theory, self.ell, self.bpwf, Nellbins=25)
    #     # print(cl_ell_theory)
    #     # print('cl gg theory: ', dl_1h_theory)
    #     print('cl gg: ', cl_gg_binned)
    #     # print('ell gg: ', ell_gg_binned)
    #     # ########
    #     # Cl_gxmu
    #     ########
    #     theory_gm = self.theory.get_Cl_gxmu()
    #     cl_ell_theory_gm = theory_gm['ell']
    #     dl_1h_theory_gm = theory_gm['1h']
    #     dl_2h_theory_gm = theory_gm['2h']
    #     ell = np.asarray(cl_ell_theory)
    #     dl_gm_theory = np.asarray(dl_1h_theory_gm) + np.asarray(dl_2h_theory_gm)
    #     ell_gm_binned, cl_gm_binned = self._bin(ell_theory, dl_gm_theory, self.ell, self.bpwf, Nellbins=40)
    #     #print('cl gm: ', cl_gm_binned)
    #     ########
    #     # Cl_muxmu
    #     ########
    #     theory_mm = self.theory.get_Cl_muxmu()
    #     cl_ell_theory_mm = theory_mm['ell']
    #     dl_1h_theory_mm = theory_mm['1h']
    #     dl_2h_theory_mm = theory_mm['2h']
    #     dl_mm_theory = np.asarray(list(dl_1h_theory_mm)) + np.asarray(list(dl_2h_theory_mm))
    #     ell_theory = np.asarray(list(cl_ell_theory_mm))
    #     ell_mm_binned, cl_mm_binned = self._bin(ell, dl_mm_theory, self.ell_full, wind_gg, Nellbins=9, ellmin = 100.5, ellmax = 1000.5)
    #     #print('cl mm: ', cl_mm_binned)
    #     #print("total cl + SN: ", cl_gg_binned + 2*(5*s-2)*cl_gm_binned + (5*s-2)*(5*s-2)*cl_mm_binned + shot_noise)
    #
    #     cl_gg = cl_gg_binned + 2*(5*s-2)*cl_gm_binned + (5*s-2)*(5*s-2)*cl_mm_binned + shot_noise
    #     #apply TF
    #     #gg = cl_gg*trans_gg
    #     #cut the first point in gg
    #     gg = gg[1:]
    #     print(gg)
    #     return gg
