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


class GXK_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    gk_data_file: Optional[str] = None
    cov_gk_data_file: Optional[str]  = None
    s_file: Optional[str]  =  None
    #print(params)
    bp_wind_file: Optional[str] = None # for binning
    pixwind_file: Optional[str] = None #healpy pixel window fucntions
    Nbins: Optional[str] = None

    # Load the templates
    def initialize(self):


        Npoints = self.Nbins
        print(Npoints)
        self.datafile = self.gk_data_file
        self.covfile = self.cov_gk_data_file

        self.pw_bin  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_file))
        self.bpwf = np.load(os.path.join(self.data_directory, self.bp_wind_file))[0]

        cov = np.loadtxt(os.path.join(self.data_directory, self.covfile))
        D = np.loadtxt(os.path.join(self.data_directory, self.datafile))
        cov=cov[1:Npoints,1:Npoints]
        self.ell = D[0,1:Npoints]
        self.ell_full = D[0,]
        self.gk = D[1,1:Npoints]
        self.sigma = D[2,1:Npoints]
        print("gk ola:", self.gk)

        # s for the lensing magnification
        self.s = np.loadtxt(os.path.join(self.data_directory, self.s_file))
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
        return {"Cl_gxkgal": {}, } #add mu terms "Cl_gxmu": {}, "Cl_muxmu": {}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell
        y_data = self.gk
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
        s=self.s
        pixwin = self.pw_bin
        Npoints = self.Nbins
        bpwf=self.bpwf[:,0,:]
        print("Healpix pixwin", pixwin)
        print("Namaster bpwf: ", bpwf)
        ########
        # Cl_gxg
        ########
        theory = self.theory.get_Cl_gxkgal()
        #print(theory)
        cl_ell_theory = theory['ell']
        dl_1h_theory = theory['1h']
        dl_2h_theory = theory['2h']
        ell = np.asarray(list(cl_ell_theory))

        dl_gk_theory = np.asarray(list(dl_1h_theory)) + np.asarray(list(dl_2h_theory))
        print('dl_gk_theory ', dl_gk_theory)
        ell_gk_binned, cl_gk_binned = self.binning(ell, dl_gk_theory, self.ell_full, bpwf, Nellbins=Npoints)
        # print(cl_ell_theory)
        # print('cl gg theory: ', dl_1h_theory)
        #vprint('cl gg: ', cl_gg_binned)
        # print('ell gg: ', ell_gg_binned)

        # # ########
        # # Cl_gxmu
        # ########
        # theory_gm = self.theory.get_Cl_gxmu()
        # cl_ell_theory_gm = theory_gm['ell']
        # dl_1h_theory_gm = theory_gm['1h']
        # dl_2h_theory_gm = theory_gm['2h']
        # ell = np.asarray(list(cl_ell_theory))
        # dl_gm_theory = np.asarray(list(dl_1h_theory_gm)) + np.asarray(list(dl_2h_theory_gm))
        # ell_gm_binned, cl_gm_binned = self.binning(ell, dl_gm_theory, self.ell_full, wind_gg, Nellbins=Npoints)
        # #print('cl gm: ', cl_gm_binned)
            #print("total cl + SN: ", cl_gg_binned + 2*(5*s-2)*cl_gm_binned + (5*s-2)*(5*s-2)*cl_mm_binned + shot_noise)
        #print("total ell: ", ell_mm_binned)
        cl_gg = cl_gg_binned #+ 2*(5*s-2)*cl_gm_binned
        # print(cl_gg)
        #apply TF
        trans = np.append(trans_gg, np.ones(Npoints-len(trans_gg)))
        gg = cl_gg *trans
        #cut the first point in gg
        gg = gg[1:]
        #print(gg)
        return gg
