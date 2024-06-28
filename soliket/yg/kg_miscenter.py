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
from scipy.integrate import simps
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class KXG_MISCENTER_Likelihood(GaussianLikelihood):
    params = {"m_shear_calibration": 0., "amplid_IA": 1.0, 'cmis': 0,}
    data_directory: Optional[str] = None
    gxk_data_file: Optional[str] = None
    cov_data_file: Optional[str] = None
    s_file: Optional[str] = None #s for lens mag
    miscenter_file: Optional[str] = None
    bp_wind_gk_file: Optional[str] = None
    pixwind_1024_file: Optional[str] = None
    Nbins_kg: Optional[str] = None
    Maglim_bin: Optional[str] = None
    # Load the data
    def initialize(self):
        self.covfile = self.cov_data_file
        self.s = np.loadtxt(os.path.join(self.data_directory, self.s_file))
        self.R, self.z = np.loadtxt(os.path.join(self.data_directory, self.miscenter_file))
        self.bpwf_kg = np.load(os.path.join(self.data_directory, self.bp_wind_gk_file))[0]
        self.pw_bin_kg  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_1024_file))
        Np_kg = self.Nbins_kg
        Npoints = Np_kg
        M = self.Maglim_bin
        # print("bin:", M)

        D_kg = np.loadtxt(os.path.join(self.data_directory, self.gxk_data_file))
        cov_full = np.loadtxt(os.path.join(self.data_directory, self.covfile))[:36, :36]

        self.ell_kg = D_kg[0,:Np_kg]
        self.ell_kg_full = D_kg[0,:Np_kg]
        self.kg = D_kg[1,:Np_kg]
        # print("ell ola kg :", self.ell_kg)
        # print("kg ola:", self.kg)
        # print("kg shape: ", self.kg.shape)

        cov = cov_full[9*(M-1):9*M, 9*(M-1):9*M]
        # print(cov)

        self.covmat =  cov
        # print("covmat shape", self.covmat.shape)
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))

        ###Combine into 1 data vector
        self.cl_joint = self.kg
        self.ell_joint = self.ell_kg
        super().initialize()

    # def get_requirements(self):
    #     return {"Cl_yxg": {}, "Cl_yxmu": {}}
    def get_requirements(self):
        return {"Cl_galnxgallens": {}, "Cl_lensmagnxgallens": {}, "Cl_galnxIA":{}}

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
    
    def _miscenter(self, Cl_orig, l_array, fmis, sigmaR_val, zbin_mean):
        # sigmaR_val = cmis * Rvir, #np.mean(self.PS_prepDV.r_vir_mat) ???????
        nl = len(l_array)
        l_array_full = np.linspace(np.min(l_array), np.max(l_array), 50000)
        nl_full = len(l_array_full)

        Cl_yg_interp = interp1d(np.log(l_array), Cl_orig)
        Cl_yg_full = Cl_yg_interp(np.log(l_array_full))
        theta_min = 1e-5
        theta_max = 0.1

        theta_array_rad = np.logspace(np.log10(theta_min), np.log10(theta_max), 40)
        ntheta = len(theta_array_rad)

        #Eq. 24
        Cl_yg_mat = (np.tile(Cl_yg_full.reshape(1, nl_full), (ntheta, 1)))
        l_theta = (np.tile(l_array_full.reshape(1, nl_full),
                       (ntheta, 1))) * (np.tile(theta_array_rad.reshape(ntheta, 1), (1, nl_full)))
        j0_ltheta = jv(0, l_theta)
        l_mat = (np.tile(l_array_full.reshape(1, nl_full), (ntheta, 1)))
        Cl_yg_theta = (simps(l_mat * Cl_yg_mat * j0_ltheta, l_array_full)) / (2 * np.pi)
        cosmo = FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc, Om0=0.3)

        R_array = np.asarray(theta_array_rad * cosmo.angular_diameter_distance(zbin_mean))

        Rmis_array = np.logspace(-4, 1, 28)
        psi_array = np.linspace(0, 2 * np.pi, 28)
        cospsi_array = np.cos(psi_array)
        nRmis = len(Rmis_array)
        npsi = len(psi_array)

        Rmis_nRmis_npsi = (np.tile(Rmis_array.reshape(1, nRmis, 1), (ntheta, 1, npsi)))
        cospsi_nRmis_npsi = (np.tile(cospsi_array.reshape(1, 1, npsi), (ntheta, nRmis, 1)))

        theta_min_rad_full = theta_min
        theta_max_rad_full = theta_max
        theta_array_rad_full = np.logspace(np.log10(theta_min_rad_full), np.log10(theta_max_rad_full), 3800)
        ntheta_full = len(theta_array_rad_full)

        Rmat_nRmis_npsi = (np.tile(R_array.reshape(ntheta, 1, 1), (1, nRmis, npsi)))

        R_arg_new = np.sqrt(
            Rmat_nRmis_npsi**2 + Rmis_nRmis_npsi**2 + 2 * Rmat_nRmis_npsi * Rmis_nRmis_npsi * cospsi_nRmis_npsi
            )
        # Eq. 25
        Cly_theta_interp = interp1d(R_array, Cl_yg_theta, fill_value=0.0, bounds_error=False)
        Cly_theta_argnew = Cly_theta_interp(R_arg_new)

        Cly_intpsi = (1. / (2 * np.pi)) * simps(Cly_theta_argnew, psi_array)

        sigmaR_mat = (np.tile(sigmaR_val, (ntheta, nRmis)))
        Rmis_mat = (np.tile(Rmis_array.reshape(1, nRmis), (ntheta, 1)))
        PRmis_mat = (Rmis_mat / sigmaR_mat**2) * np.exp(-1. * ((Rmis_mat**2) / (2. * sigmaR_mat**2)))

        #Eq. 27
        Cly_intRmis = simps(Cly_intpsi * PRmis_mat, Rmis_array)

        if np.all(Cly_intRmis > 0):
            Cly_intRmis_interp = interp1d(np.log(theta_array_rad), np.log(Cly_intRmis))
            Cly_misc_theta = np.exp(Cly_intRmis_interp(np.log(theta_array_rad_full)))
        else:
            # print 'negative values in Cly_intRmis. Careful about extrapolation!!!!'
            Cly_intRmis_interp = interp1d(np.log(theta_array_rad), Cly_intRmis)
            Cly_misc_theta = (Cly_intRmis_interp(np.log(theta_array_rad_full)))

        Cly_misc_theta_full = np.tile(Cly_misc_theta.reshape(1, ntheta_full), (nl, 1))
        l_thetafull = (np.tile(l_array.reshape(nl, 1),
                               (1, ntheta_full))) * (np.tile(theta_array_rad_full.reshape(1, ntheta_full), (nl, 1)))
        j0_lthetafull = jv(0, l_thetafull)
        theta_mat = (np.tile(theta_array_rad_full.reshape(1, ntheta_full), (nl, 1)))
        Cly_misc_l = (2 * np.pi
                     ) * simps(theta_mat * Cly_misc_theta_full * j0_lthetafull, theta_array_rad_full)

        # Cly_origcheck_theta_full = np.tile(Cly_origcheck_theta.reshape(1, ntheta_full), (nl, 1))
        # Cly_origcheck_l = (2 * np.pi) * (
        #     sp.integrate.simps(theta_mat * Cly_origcheck_theta_full * j0_lthetafull, theta_array_rad_full))
        Cly_misc_l_final = fmis * Cly_misc_l + (1 - fmis) * Cl_orig

        return l_array, Cly_misc_l_final

    def _cl2dl(self, l):
        return l*(l+1)/2/np.pi
    
    def _get_theory(self, **params_values_dict):
        alpha=self.s
        m = params_values_dict['m_shear_calibration']
        A_IA = params_values_dict['amplid_IA']
        bpwf_kg = self.bpwf_kg[:,0,:]
        pixwin_kg = self.pw_bin_kg
        Np_kg = self.Nbins_kg
        ellmax_bin_kg = 2200
        fmis = 1.0
        cmis = params_values_dict['cmis']
        # print("cmis", cmis)
        Rvir = self.R
        zbin_mean = self.z
        # print(Rvir, zbin_mean)

        kg_1h_all, kg_2h_all, km_all, IA_all,  kg_1h_all_miscenter = [], [], [], [], []
        theory_kg = self.provider.get_Cl_galnxgallens()
        theory_km = self.provider.get_Cl_lensmagnxgallens()
        theory_gIA = self.provider.get_Cl_galnxIA()
        for i in range(2):
            Nb=str(i)
            ell_theory_kg, cl_1h_theory_kg, cl_2h_theory_kg = theory_kg[Nb]['ell'], theory_kg[Nb]['1h'], theory_kg[Nb]['2h']
            ell_theory_km, cl_1h_theory_km, cl_2h_theory_km = theory_km[Nb]['ell'], theory_km[Nb]['1h'], theory_km[Nb]['2h']
            ell_theory_gIA,  cl_2h_theory_gIA = theory_gIA[Nb]['ell'],  theory_gIA[Nb]['2h']

            # print("cl_1h_theory_kg:", cl_1h_theory_kg[:10])
            # print("cl_2h_theory_kg:", cl_2h_theory_kg[:10])
            ell_kg_bin, dl_kg_bin_1h = self._bin(ell_theory_kg, np.asarray(cl_1h_theory_kg), self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
            ell_kg_bin, dl_kg_bin_2h = self._bin(ell_theory_kg, np.asarray(cl_2h_theory_kg), self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
            ell_km_bin, dl_km_bin = self._bin(ell_theory_km, np.asarray(cl_1h_theory_km) + np.asarray(cl_2h_theory_km), self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
            ell_gIA_bin, dl_gIA_bin = self._bin(ell_theory_gIA, np.asarray(cl_2h_theory_gIA), self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
            # print("dl_kg_bin:", dl_kg_bin_1h+dl_kg_bin_2h)
            # print("dl_km_bin:", dl_km_bin)
            # print("dl_gIA_bin:", dl_gIA_bin

            ### Miscenter
            #First bin, then miscenter (do it for all 8 bins)
            sigmaR_val = cmis * Rvir
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            #ell_kg_bin, dl_kg_bin_1h_mis = self._bin(ell_theory_kg, cl_1h_theory_kg , self.ell_kg_full, ellmax_bin_kg, bpwf_kg, pixwin_kg, Nellbins=Np_kg, conv2cl=True)
            ell_miscenter_kg, dl_kg_bin_1h_miscenter = self._miscenter(dl_kg_bin_1h/self._cl2dl(ell_kg_bin), ell_kg_bin, fmis, sigmaR_val, zbin_mean)
            ## Append
            kg_1h_all.append(dl_kg_bin_1h), kg_2h_all.append(dl_kg_bin_2h), km_all.append(dl_km_bin), IA_all.append(dl_gIA_bin)
            kg_1h_all_miscenter.append(dl_kg_bin_1h_miscenter*self._cl2dl(ell_miscenter_kg))


        # print("diff kg = ", (kg_1h_all[0] - kg_1h_all[1])/kg_1h_all[0])
        kg_1h_cen_mis = kg_1h_all_miscenter[1]
        kg_1h_sat = kg_1h_all[0] - kg_1h_all[1]
        # print("1h kg cent mis: ", kg_1h_cen_mis)
        
        kg = (1+m)*(kg_1h_cen_mis+kg_1h_sat +kg_2h_all[0] + 2*(alpha-1)*km_all[0] + A_IA*IA_all[0]) # shear calibration m

        # print("cl joint:",kg)

        if np.isnan(kg).any()==True:
            print("Nans in the theory prediction!")
            exit()
        return kg
