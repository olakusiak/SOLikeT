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

class YXG_ALLBINS_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    y_map: Optional[str] = None
    yxg_data_file: Optional[str] = None
    cov_data_file: Optional[str] = None
    bp_wind_yg_file: Optional[str] = None
    pixwind_4096_file: Optional[str] = None
    Nbins_yg: Optional[str] = None
    # Load the data
    def initialize(self):
        self.covmat = np.loadtxt(os.path.join(self.data_directory, self.cov_data_file))
        self.bpwf_yg = np.load(os.path.join(self.data_directory, self.bp_wind_yg_file))[0]
        self.pw_bin_yg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_4096_file))
        Np_yg = self.Nbins_yg
        Npoints = Np_yg
        Nbins = 4

        Cl_yg_all = []
        Sig_yg_all = []
        for i in range(1, Nbins+1):
            D_yg = np.loadtxt(self.data_directory + self.yxg_data_file + str(i) +"_dl.txt")
            self.ell_yg = D_yg[0,:Np_yg]
            self.ell_yg_full = D_yg[0,:Np_yg]
            self.yg = D_yg[1,:Np_yg]
            # self.sigma_yg = D_yg[2,:Np_yg]

            # print("ell ola yg :", self.ell_yg)
            # print("yg ola:", self.yg)
            Cl_yg_all.append(D_yg[1,:Np_yg])
            Sig_yg_all.append(D_yg[2,:Np_yg])


        # Sig_all = np.concatenate((np.concatenate((Cl_yg_all)),np.concatenate((Cl_kg_all))), axis=0)
        #self.covmat = np.diag(Sig_all**2)
        Nkg = 9*4
        self.covmat =  self.covmat[Nkg:,Nkg:]
        print("self.covmat:", (self.covmat).shape)
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        #print("cov:", (self.covmat).shape)

        # Combine all data into one data vector
        self.cl_joint = np.concatenate((Cl_yg_all), axis=0)
        self.ell_joint = np.concatenate((self.ell_yg, self.ell_yg, self.ell_yg, self.ell_yg,), axis=0)
        # print("self.ell_joint:", self.ell_joint)
        # print("self.cl_joint:", self.cl_joint)
        super().initialize()

    # def get_requirements(self):
    #     return {"Cl_yxg": {}, "Cl_yxmu": {}}
    def get_requirements(self):
        return {"Cl_galnxtsz": {}, "Cl_lensmagnxtsz": {}}

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
        f_int =  interp1d(ell_theory, cl_theory, fill_value="extrapolate")
        inter_cl = np.asarray(f_int(new_ell))
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

    def _get_theory(self, **params_values):
        alpha_lens_mag_list=[1.21, 1.15, 1.88, 1.97]
        bpwf_yg = self.bpwf_yg[:,0,:]
        pixwin_yg = self.pw_bin_yg
        Np_yg = self.Nbins_yg
        ellmax_bin_yg = 5600

        yg_all = []
        theory_yg = self.theory.get_Cl_galnxtsz()
        theory_ym = self.theory.get_Cl_lensmagnxtsz()
        # print(theory_kg)

        for i in range(len(theory_yg)):
            Nb=str(i)
            ell_theory_yg, cl_1h_theory_yg, cl_2h_theory_yg = theory_yg[Nb]['ell'], theory_yg[Nb]['1h'], theory_yg[Nb]['2h']
            ell_theory_ym, cl_1h_theory_ym, cl_2h_theory_ym = theory_ym[Nb]['ell'], theory_ym[Nb]['1h'], theory_ym[Nb]['2h']
            # print("ell_theory_yg",ell_theory_yg)
            # print("cl_1h_theory_yg:", cl_1h_theory_yg[:10])
            # print("cl_2h_theory_yg:", cl_2h_theory_yg[:10])
            # dl_theory_yg = np.asarray(cl_1h_theory_yg) + np.asarray(cl_2h_theory_yg)
            ell_yg_bin, dl_yg_bin = self._bin(ell_theory_yg, np.asarray(cl_1h_theory_yg) + np.asarray(cl_2h_theory_yg), self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg, Nellbins=Np_yg, conv2cl=True)
            ell_ym_bin, dl_ym_bin = self._bin(ell_theory_ym, np.asarray(cl_1h_theory_ym) + np.asarray(cl_2h_theory_ym), self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg, Nellbins=Np_yg, conv2cl=True)
            # print("dl_yg_bin:", dl_yg_bin[:10])
            # print("dl_ym_bin:", dl_ym_bin[:10])
            alpha = alpha_lens_mag_list[i]
            yg = 1.e-6*(dl_yg_bin + 2*(alpha-1)*dl_ym_bin)
            # print("yg: ", yg[:10])
            yg_all.append(yg)

        cl_joint = np.concatenate((yg_all), axis=0)

        if np.isnan(cl_joint).any()==True:
            print("Nans in the theory prediction!")
            exit()
        # print("cl joint:", yg_all)
        return cl_joint
