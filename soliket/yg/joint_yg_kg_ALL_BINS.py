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

class YXG_KXG_ALLBINS_Likelihood(GaussianLikelihood):
    data_directory: Optional[str] = None
    y_map: Optional[str] = None
    params = {"m_shear_calibration": 0., "amplid_IA": 1.0}
    yxg_data_file: Optional[str] = None
    gxk_data_file: Optional[str] = None
    cov_data_file: Optional[str] = None
    bp_wind_yg_file: Optional[str] = None
    bp_wind_gk_file: Optional[str] = None
    pixwind_4096_file: Optional[str] = None
    pixwind_1024_file: Optional[str] = None
    Nbins_yg: Optional[str] = None
    Nbins_kg: Optional[str] = None
    # Load the data
    def initialize(self):
        self.covmat = np.loadtxt(os.path.join(self.data_directory, self.cov_data_file))
        self.bpwf_yg = np.load(os.path.join(self.data_directory, self.bp_wind_yg_file))[0]
        self.bpwf_kg = np.load(os.path.join(self.data_directory, self.bp_wind_gk_file))[0]
        self.pw_bin_yg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_4096_file))
        self.pw_bin_kg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_1024_file))
        Np_yg = self.Nbins_yg
        Np_kg = self.Nbins_kg
        Npoints = Np_kg + Np_yg
        Nbins = 4

        Cl_yg_all, Cl_kg_all = [], []
        Sig_yg_all, Sig_kg_all = [], []
        for i in range(1, Nbins+1):
            # print("Maglim", i)
            D_yg = np.loadtxt(self.data_directory + self.yxg_data_file + str(i) +"_dl.txt")
            D_kg = np.loadtxt(self.data_directory + self.gxk_data_file + str(int(i)) + "_kappa4_dl.txt")
            self.ell_yg = D_yg[0,:Np_yg]
            self.ell_yg_full = D_yg[0,:Np_yg]
            self.yg = D_yg[1,:Np_yg]
            self.sigma_yg = D_yg[2,:Np_yg]

            self.ell_kg = D_kg[0,:Np_kg]
            self.ell_kg_full = D_kg[0,:Np_kg]
            self.kg = D_kg[1,:Np_kg]
            self.sigma_kg = D_kg[2,:Np_kg]
            Cl_yg_all.append(D_yg[1,:Np_yg])
            Cl_kg_all.append(D_kg[1,:Np_kg])
            Sig_yg_all.append(D_yg[2,:Np_yg])
            Sig_kg_all.append(D_kg[2,:Np_kg])


        # Sig_all = np.concatenate((np.concatenate((Cl_yg_all)),np.concatenate((Cl_kg_all))), axis=0)
        # self.covmat = np.diag(Sig_all**2)
        # self.covmat =  cov[:Npoints,:Npoints]
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        #print("cov:", (self.covmat).shape)
        #print("Npoints = ", Npoints*4)

        # Combine all data into one data vector
        self.cl_joint = np.concatenate((np.concatenate((Cl_kg_all)),np.concatenate((Cl_yg_all))), axis=0)
        self.ell_joint = np.concatenate((self.ell_kg, self.ell_kg,self.ell_kg, self.ell_kg, self.ell_yg, self.ell_yg, self.ell_yg, self.ell_yg,), axis=0)
        # print("self.ell_joint:", self.ell_joint)
        #print("self.cl_joint:", self.cl_joint)
        super().initialize()

    # def get_requirements(self):
    #     return {"Cl_yxg": {}, "Cl_yxmu": {}}
    def get_requirements(self):
        return {"Cl_galnxtsz": {}, "Cl_galnxgallens": {},"Cl_lensmagnxtsz": {}, "Cl_lensmagnxgallens": {}, "Cl_galnxIA":{}}

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

    def _get_theory(self, **params_values_dict):
        alpha_lens_mag_list=[1.21, 1.15, 1.88, 1.97]
        m = params_values_dict['m_shear_calibration']
        A_IA = params_values_dict['amplid_IA']
        bpwf_yg = self.bpwf_yg[:,0,:]
        bpwf_kg = self.bpwf_kg[:,0,:]
        pixwin_yg = self.pw_bin_yg
        pixwin_kg = self.pw_bin_kg
        Np_yg = self.Nbins_yg
        Np_kg = self.Nbins_kg
        ellmax_bin_kg = 2200
        ellmax_bin_yg = 5600

        yg_all, kg_all = [], []
        theory_yg = self.theory.get_Cl_galnxtsz()
        theory_kg = self.theory.get_Cl_galnxgallens()
        theory_ym = self.theory.get_Cl_lensmagnxtsz()
        theory_km = self.theory.get_Cl_lensmagnxgallens()
        theory_gIA = self.theory.get_Cl_galnxIA()
        # print(theory_kg)

        for i in range(len(theory_yg)):
            Nb=str(i)
            ell_theory_yg, cl_1h_theory_yg, cl_2h_theory_yg = theory_yg[Nb]['ell'], theory_yg[Nb]['1h'], theory_yg[Nb]['2h']
            ell_theory_kg, cl_1h_theory_kg, cl_2h_theory_kg = theory_kg[Nb]['ell'], theory_kg[Nb]['1h'], theory_kg[Nb]['2h']
            ell_theory_ym, cl_1h_theory_ym, cl_2h_theory_ym = theory_ym[Nb]['ell'], theory_ym[Nb]['1h'], theory_ym[Nb]['2h']
            ell_theory_km, cl_1h_theory_km, cl_2h_theory_km = theory_km[Nb]['ell'], theory_km[Nb]['1h'], theory_km[Nb]['2h']
            ell_theory_gIA,  cl_2h_theory_gIA = theory_gIA[Nb]['ell'],  theory_gIA[Nb]['2h']
            # print("ell_theory_yg",ell_theory_yg)
            # print("cl_1h_theory_kg:", cl_1h_theory_kg[:10])
            # print("cl_2h_theory_kg:", cl_2h_theory_kg[:10])
            # # dl_theory_yg = np.asarray(cl_1h_theory_yg) + np.asarray(cl_2h_theory_yg)
            ell_yg_bin, dl_yg_bin = self._bin(ell_theory_yg, np.asarray(cl_1h_theory_yg) + np.asarray(cl_2h_theory_yg), self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg, Nellbins=Np_yg, conv2cl=True)
            ell_kg_bin, dl_kg_bin = self._bin(ell_theory_kg, np.asarray(cl_1h_theory_kg) + np.asarray(cl_2h_theory_kg), self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
            ell_ym_bin, dl_ym_bin = self._bin(ell_theory_ym, np.asarray(cl_1h_theory_ym) + np.asarray(cl_2h_theory_ym), self.ell_yg_full, ellmax_bin_yg, bpwf_yg, pixwin_yg, Nellbins=Np_yg, conv2cl=True)
            ell_km_bin, dl_km_bin = self._bin(ell_theory_km, np.asarray(cl_1h_theory_km) + np.asarray(cl_2h_theory_km), self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
            ell_gIA_bin, dl_gIA_bin = self._bin(ell_theory_gIA, np.asarray(cl_2h_theory_gIA), self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
            # print("dl_kg_bin:", dl_kg_bin)
            # print("dl_km_bin:", dl_km_bin)
            # print("dl_gIA_bin:", dl_gIA_bin)
            alpha = alpha_lens_mag_list[i]
            kg = (1+m)*(dl_kg_bin + 2*(alpha-1)*dl_km_bin + A_IA*dl_gIA_bin) # shear calibration m
            yg = 1.e-6*(dl_yg_bin + 2*(alpha-1)*dl_ym_bin)
            # print("yg: ", yg[:10])
            # print("kg: ", kg)
            # print("yg + kg shape",yg.shape+kg.shape)
            yg_all.append(yg)
            kg_all.append(kg)
        # print("kg: ", kg_all)
        # print("yg: ", yg_all)

        cl_joint = np.concatenate((np.concatenate(kg_all), np.concatenate(yg_all)), axis=0)
        #print("cl joint:", cl_joint)

        if np.isnan(cl_joint).any()==True:
            print("Nans in the theory prediction!")
            exit()
        return cl_joint
