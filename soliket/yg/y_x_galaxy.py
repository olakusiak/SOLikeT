"""
.. module:: yxg

:Synopsis: Definition of simplistic y-map power spectrum likelihood using pre-computed frequency dependent  templates
           for yxg powerspectrum (A_yxg), and foregrounds, e.g., A_cib, A_ir, A_rs, A_cn.

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



class YXG_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    yxg_data_file: Optional[str] = None
    cov_yxg_data_file: Optional[str] = None
    s_file: Optional[str] = None #s for lens mag
    bp_wind_file: Optional[str] = None

    # Load the data
    def initialize(self):
        self.datafile = self.yxg_data_file
        self.covfile = self.cov_yxg_data_file
        self.s = np.loadtxt(os.path.join(self.data_directory, self.s_file))
        self.bpwf = np.load(os.path.join(self.data_directory, self.bp_wind_file))[0]

        D = np.loadtxt(os.path.join(self.data_directory, self.datafile))
        cov = np.loadtxt(os.path.join(self.data_directory, self.covfile))

        self.ell = D[0,1:]
        self.yg = D[1,1:]
        self.sigma_tot = D[2,1:]

        #self.cvg = np.asarray(self.sigma_tot)**2.
        #self.cvg = np.diag(self.cvg)
        self.cvg = cov[1:, 1:]
        self.covmat = self.cvg
        print("ell ola:", self.ell)
        print("yg ola:", self.yg)

        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        #print('[reading ref yxg data files] read-in completed.')
        super().initialize()


    def get_requirements(self):
        return {"Cl_yxg": {}, "Cl_yxmu": {}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell
        y_data = self.yg
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov

    def _bin(self, ell_theory, cl_theory, ell_data, bpwf, Nellbins=40, conv2cl=True,):
        """
        Interpolate the theory dl's, and bin according to the bandpower window function (bpwf)
        """
        #interpolate
        new_ell = np.arange(2, 8000, 1)
        cl_theory_log = np.log(cl_theory)
        f_int =  interp1d(ell_theory, cl_theory_log)
        inter_cl_log = np.asarray(f_int(new_ell))
        inter_cl= np.exp(inter_cl_log)
        if conv2cl==True: #go from dls to cls because the bpwf mutliplies by ell*(ell+1)/2pi
            inter_cl= inter_cl*(2.0*np.pi)/(new_ell)/(new_ell+1.0)

        #bin according to the bpwf
        cl_binned = np.zeros(Nellbins)
        for i in range (Nellbins):
            wi = bpwf[i]
            # wi starts from ell=2 according to Alex, email 1-9-22; could add ell=0,1, but would contribute nothing to the sum
            cl_binned[i] = np.sum(wi[2:len(inter_cl)+2]*inter_cl)
        #print("clbinned:", cl_binned)
        return ell_data, cl_binned


##PIXWIN pls
    def _get_theory(self, **params_values):
        s=self.s
        bpwf=self.bpwf[:,0,:]
        print("bpwf: ", bpwf)
        # ########
        # Cl_yxg
        ########
        theory_yg = self.theory.get_Cl_yxg()
        ell_theory_yg = theory_yg['ell']
        dl_1h_theory_yg = theory_yg['1h']
        dl_2h_theory_yg = theory_yg['2h']
        #print("cl_1h_theory_yg:", cl_1h_theory_yg)
        dl_theory_yg = np.asarray(list(dl_1h_theory_yg)) + np.asarray(list(dl_2h_theory_yg))
        ell_yg_bin, dl_yg_bin = self._bin(ell_theory_yg, dl_theory_yg, self.ell, bpwf, Nellbins=40, conv2cl=True)
        print("yg bin: ", ell_yg_bin, dl_yg_bin)

        # ########
        # Cl_yxmu
        ########
        theory_ym = self.theory.get_Cl_yxmu()
        ell_theory_ym = theory_ym['ell']
        cl_1h_theory_ym = theory_ym['1h']
        cl_2h_theory_ym = theory_ym['2h']
        dl_theory_ym = np.asarray((cl_1h_theory_ym)) + np.asarray((cl_2h_theory_ym))
        ell_ym_bin, dl_ym_bin =  self._bin(ell_theory_ym, dl_theory_ym, self.ell, bpwf, Nellbins=40, conv2cl=True)

        #print("ym bin: ", ell_ym_bin, dl_ym_bin)
        #rint("yg:", 1e-6*(dl_yg_bin+(5*s-2)*dl_ym_bin))

        # unit conversion:
        return 1e-6*(dl_yg_bin+(5*s-2)*dl_ym_bin) #1e-6*(dl_yg_bin+(5-2s)*dl_ym_bin)
