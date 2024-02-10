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

class YXG_KXG_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    yxg_data_file: Optional[str] = None
    gxk_data_file: Optional[str] = None
    cov_data_file: Optional[str] = None
    s_file: Optional[str] = None #s for lens mag
    bp_wind_yg_file: Optional[str] = None
    bp_wind_gk_file: Optional[str] = None
    pixwind_4096_file: Optional[str] = None
    pixwind_1024_file: Optional[str] = None
    Nbins_yg: Optional[str] = None
    Nbins_kg: Optional[str] = None
    # Load the data
    def initialize(self):
        self.covfile = self.cov_data_file
        self.s = np.loadtxt(os.path.join(self.data_directory, self.s_file))
        self.bpwf_yg = np.load(os.path.join(self.data_directory, self.bp_wind_yg_file))[0]
        self.bpwf_kg = np.load(os.path.join(self.data_directory, self.bp_wind_gk_file))[0]
        self.pw_bin_yg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_4096_file))
        self.pw_bin_kg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_1024_file))
        Np_yg = self.Nbins_yg
        Np_kg = self.Nbins_kg

        D_yg = np.loadtxt(os.path.join(self.data_directory, self.yxg_data_file))
        D_kg = np.loadtxt(os.path.join(self.data_directory, self.gxk_data_file))
        cov = np.loadtxt(os.path.join(self.data_directory, self.covfile))

        self.ell_yg = D_yg[0,:Np_yg]
        self.ell_yg_full = D_yg[0,:Np_yg]
        self.yg = D_yg[1,:Np_yg]
        self.sigma_yg = D_yg[2,:Np_yg]

        self.ell_kg = D_kg[0,:Np_kg]
        self.ell_kg_full = D_kg[0,:Np_kg]
        self.kg = D_kg[1,:Np_kg]
        self.sigma_kg = D_kg[2,:Np_kg]
        # print("ell ola yg :", self.ell_yg)
        # print("yg ola:", self.yg)
        # # print("yg shape: ", self.yg.shape)
        # print("ell ola kg :", self.ell_kg)
        # print("kg ola:", self.kg)
        # # print("kg shape: ", self.kg.shape)
        # #
        # print("cov shape",cov.shape)
        Npoints = Np_kg + Np_yg

        self.covmat =  cov[:Npoints,:Npoints]
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))

        ###Combine into 1 data vector
        self.cl_joint = np.concatenate((self.kg, self.yg), axis=0)
        self.ell_joint = np.concatenate((self.ell_kg, self.ell_yg), axis=0)
        super().initialize()

    # def get_requirements(self):
    #     return {"Cl_yxg": {}, "Cl_yxmu": {}}
    def get_requirements(self):
        return {"Cl_yxg": {}, "Cl_yxmu": {},"Cl_kgxg": {}, "Cl_IAxg": {}, "Cl_kgxmu":{}}

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
        f_int =  interp1d(ell_theory, cl_theory_log)
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

    def _get_theory(self, **params_values):
        alpha=self.s
        bpwf_yg = self.bpwf_yg[:,0,:]
        bpwf_kg = self.bpwf_kg[:,0,:]
        pixwin_yg = self.pw_bin_yg
        pixwin_kg = self.pw_bin_kg
        Np_yg = self.Nbins_yg
        Np_kg = self.Nbins_kg
        ellmax_bin_kg = 2200
        ellmax_bin_yg = 5600

        # ########
        # Cl_yxg
        ########
        theory_yg = self.theory.get_Cl_yxg()
        ell_theory_yg = theory_yg['ell']
        cl_1h_theory_yg = theory_yg['1h']
        cl_2h_theory_yg = theory_yg['2h']
        #print("ell_theory_yg",ell_theory_yg)
        #print("cl_1h_theory_yg:", cl_1h_theory_yg[:10])
        #print("cl_2h_theory_yg:", cl_2h_theory_yg[:10])
        dl_theory_yg = np.asarray(list(cl_1h_theory_yg)) + np.asarray(list(cl_2h_theory_yg))
        ell_yg_bin, dl_yg_bin = self._bin(ell_theory_yg, dl_theory_yg, self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg, Nellbins=Np_yg, conv2cl=True)
        # print("ell_yg_bin: ", ell_yg_bin)
        #print("yg bin: ", dl_yg_bin[:10])

        # ########
        # Cl_yxmu
        ########
        theory_ym = self.theory.get_Cl_yxmu()
        ell_theory_ym = theory_ym['ell']
        cl_1h_theory_ym = theory_ym['1h']
        cl_2h_theory_ym = theory_ym['2h']
        #print("cl_1h_theory_ym:", cl_1h_theory_ym[:10])
        #print("cl_2h_theory_ym:", cl_2h_theory_ym[:10])
        dl_theory_ym = np.asarray((cl_1h_theory_ym)) + np.asarray((cl_2h_theory_ym))
        ell_ym_bin, dl_ym_bin =  self._bin(ell_theory_ym, dl_theory_ym, self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg, Nellbins=Np_yg, conv2cl=True)

        ########
        # Cl_kgxg
        ########
        theory_kg = self.theory.get_Cl_kgxg()
        ell_theory_kg = theory_kg['ell']
        dl_1h_theory_kg = theory_kg['1h']
        dl_2h_theory_kg = theory_kg['2h']
        dl_gk_theory = np.asarray(list(dl_1h_theory_kg)) + np.asarray(list(dl_2h_theory_kg))
        #print('ell gk_theory ', ell_theory_kg)
        #print('dl_gk_theory ', dl_gk_theory)
        ell_gk_bin, cl_gk_bin = self._bin(ell_theory_kg, dl_gk_theory, self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
        # print('ell_gk_bin: ', ell_gk_bin)
        # print('cl gk theory: ', dl_1h_theory_kg)
        #print('cl kg: ', cl_gk_bin)

        # ########
        # Cl_kgxmu
        ########
        theory_km = self.theory.get_Cl_kgxmu()
        ell_theory_km = theory_km['ell']
        dl_1h_theory_km = theory_km['1h']
        dl_2h_theory_km = theory_km['2h']
        dl_km_theory = np.asarray(list(dl_1h_theory_km)) + np.asarray(list(dl_2h_theory_km))
        ell_km_bin, cl_km_bin = self._bin(ell_theory_km, dl_km_theory, self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
        #print('cl gm: ', cl_km_bin)

        #########
        # Cl_IAxg
        ########
        theory_IA = self.theory.get_Cl_IAxg()
        ell_theory_IA = theory_IA['ell']
        dl_2h_theory_IA = theory_IA['2h']
        ell_IA_bin, cl_IA_bin = self._bin(ell_theory_km, dl_2h_theory_IA, self.ell_kg, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
        # print("cl_IA_2h: ", cl_IA_bin)

        kg = cl_gk_bin + 2*(alpha-1)*cl_km_bin - cl_IA_bin
        yg = 1.e-6*(dl_yg_bin + 2*(alpha-1)*dl_ym_bin)
        # print("yg: ", yg[:10])
        # print("kg: ", kg)
        #print("yg + kg shape",yg.shape+kg.shape)

        cl_joint = np.concatenate((kg, yg), axis=0) #remove the first bin ell=50
        #print("cl joint:", cl_joint)
        return cl_joint


    # # def get_requirements(self):
    # #     return {"Cl_yxg": {}, "Cl_yxmu": {}}
    # def get_requirements(self):
    #     return {"Cl_yxg": {}, "Cl_yxmu": {},"Cl_kgxg": {}, "Cl_kgxmu": {}, "Cl_IAxg": {}}
    #
    # # this is the data to fit
    # def _get_data(self):
    #     x_data = self.ell_joint
    #     y_data = self.cl_joint
    #     return x_data, y_data
    #
    # def _get_cov(self):
    #     cov = self.covmat
    #     return cov
    # def _bin(self, ell_theory, cl_theory, ell_data, ellmax, bpwf, pix_win, Nellbins, conv2cl=True,):
    #     """
    #     Interpolate the theory dl's, and bin according to the bandpower window function (bpwf)
    #     """
    #     #interpolate
    #     # ellmax=int(np.round(ell_data[len(ell_data)-1]))
    #     # print("ellmax",ellmax)
    #     new_ell = np.arange(2, ellmax, 1)
    #     cl_theory_log = np.log(cl_theory)
    #     f_int =  interp1d(ell_theory, cl_theory_log)
    #     inter_cl_log = np.asarray(f_int(new_ell))
    #     inter_cl= np.exp(inter_cl_log)
    #     if conv2cl==True: #go from dls to cls because the bpwf mutliplies by ell*(ell+1)/2pi
    #         inter_cl= inter_cl*(2.0*np.pi)/(new_ell)/(new_ell+1.0)
    #
    #     #multiply by the pixel window function (from healpix for given nside)
    #     inter_cl = inter_cl*(pix_win[2:ellmax])**2
    #     #bin according to the bpwf
    #     cl_binned = np.zeros(Nellbins)
    #     for i in range (Nellbins):
    #         wi = bpwf[i]
    #         # wi starts from ell=2 according to Alex, email 1-9-22; could add ell=0,1, but would contribute nothing to the sum
    #         cl_binned[i] = np.sum(wi[2:len(inter_cl)+2]*inter_cl)
    #     #print("clbinned:", cl_binned)
    #     return ell_data, cl_binned
    #
    # def _get_theory(self, **params_values):
    #     alpha=self.s
    #     bpwf_yg = self.bpwf_yg[:,0,:]
    #     bpwf_kg = self.bpwf_kg[:,0,:]
    #     pixwin_yg = self.pw_bin_yg
    #     pixwin_kg = self.pw_bin_kg
    #     Np_yg = self.Nbins_yg
    #     Np_kg = self.Nbins_kg
    #     ellmax_bin_kg = 2200
    #     ellmax_bin_yg = 6000
    #
    #     # ########
    #     # Cl_yxg
    #     ########
    #     theory_yg = self.theory.get_Cl_yxg()
    #     ell_theory_yg = theory_yg['ell']
    #     cl_1h_theory_yg = theory_yg['1h']
    #     cl_2h_theory_yg = theory_yg['2h']
    #     #print("cl_1h_theory_yg:", cl_1h_theory_yg[:10])
    #     #print("cl_2h_theory_yg:", cl_2h_theory_yg[:10])
    #     dl_theory_yg = np.asarray(list(cl_1h_theory_yg)) + np.asarray(list(cl_2h_theory_yg))
    #     ell_yg_bin, dl_yg_bin = self._bin(ell_theory_yg, dl_theory_yg, self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg, Nellbins=Np_yg, conv2cl=True)
    #     #print("yg bin: ", dl_yg_bin[:10])
    #
    #     # ########
    #     # Cl_yxmu
    #     ########
    #     theory_ym = self.theory.get_Cl_yxmu()
    #     ell_theory_ym = theory_ym['ell']
    #     cl_1h_theory_ym = theory_ym['1h']
    #     cl_2h_theory_ym = theory_ym['2h']
    #     #print("cl_1h_theory_ym:", cl_1h_theory_ym[:10])
    #     #print("cl_2h_theory_ym:", cl_2h_theory_ym[:10])
    #     dl_theory_ym = np.asarray((cl_1h_theory_ym)) + np.asarray((cl_2h_theory_ym))
    #     ell_ym_bin, dl_ym_bin =  self._bin(ell_theory_ym, dl_theory_ym, self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg, Nellbins=Np_yg, conv2cl=True)
    #
    #     ########
    #     # Cl_kgxg
    #     ########
    #     theory_kg = self.theory.get_Cl_kgxg()
    #     ell_theory_kg = theory_kg['ell']
    #     dl_1h_theory_kg = theory_kg['1h']
    #     dl_2h_theory_kg = theory_kg['2h']
    #     dl_gk_theory = np.asarray(list(dl_1h_theory_kg)) + np.asarray(list(dl_2h_theory_kg))
    #     #print('dl_gk_theory ', dl_gk_theory)
    #     ell_gk_bin, cl_gk_bin = self._bin(ell_theory_kg, dl_gk_theory, self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
    #     # print('cl gk theory: ', dl_1h_theory_kg)
    #     #print('cl kg: ', cl_gk_bin)
    #
    #     # ########
    #     # Cl_kgxmu
    #     ########
    #     theory_km = self.theory.get_Cl_kgxmu()
    #     ell_theory_km = theory_km['ell']
    #     dl_1h_theory_km = theory_km['1h']
    #     dl_2h_theory_km = theory_km['2h']
    #     dl_km_theory = np.asarray(list(dl_1h_theory_km)) + np.asarray(list(dl_2h_theory_km))
    #     ell_km_bin, cl_km_bin = self._bin(ell_theory_km, dl_km_theory, self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
    #     #print('cl gm: ', cl_km_bin)
    #
    #     #########
    #     # Cl_IAxg
    #     ########
    #     theory_IA = self.theory.get_Cl_IAxg()
    #     ell_theory_IA = theory_IA['ell']
    #     dl_2h_theory_IA = theory_IA['2h']
    #     ell_IA_bin, cl_IA_bin = self._bin(ell_theory_km, dl_2h_theory_IA, self.ell_kg, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
    #     #print("cl_IA_2h: ", cl_IA_bin)
    #
    #     kg = cl_gk_bin + 2*(alpha-1)*cl_km_bin - cl_IA_bin
    #     yg = 1.e-6*(dl_yg_bin + 2*(alpha-1)*dl_ym_bin)
    #     #print("yg: ", yg)
    #     #print("kg: ", kg)
    #     #print("yg + kg shape",yg.shape+kg.shape)
    #
    #     cl_joint = np.concatenate((kg[1:], yg[1:]), axis=0) #remove the first bin ell=50
    #     #print("cl joint:", cl_joint)
    #     return cl_joint
