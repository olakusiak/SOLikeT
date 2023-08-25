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
    cov_data_file: Optional[str]  = None
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
        self.covfile = self.cov_data_file
        self.s = np.loadtxt(os.path.join(self.data_directory, self.s_file))
        self.pw_bin  = np.loadtxt(os.path.join(self.data_directory, self.pixwind_file))
        self.bpwf = np.load(os.path.join(self.data_directory, self.bp_wind_file))[0]

        D = np.loadtxt(os.path.join(self.data_directory, self.datafile))
        cov = np.loadtxt(os.path.join(self.data_directory, self.covfile))

        self.ell_full = D[0,]
        self.ell = D[0,1:Npoints]
        self.gk = D[1,1:Npoints]
        #self.sigma = D[2,:Npoints]
        #self.covmat =  cov[1:Npoints,1:Npoints]
        self.covmat =  cov[:9,:9]
        print("ell ola:", self.ell)
        print("gk ola:", self.gk)


        # now compute the iverse and det of covariance matrix
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print("eig:", np.linalg.eig(self.covmat))
        super().initialize()


    def get_requirements(self):
        return {"Cl_kgxg": {}, "Cl_kgxmu": {}, "Cl_IAxg": {}} #add mu terms "Cl_gxmu": {}, "Cl_muxmu": {}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell
        y_data = self.gk
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
        s=self.s
        pixwin = self.pw_bin
        Npoints = self.Nbins
        bpwf=self.bpwf[:,0,:]
        print("Npoints:", Npoints)
        print("s:",s)
        #print("Healpix pixwin", pixwin)
        #print("Namaster bpwf: ", bpwf)
        ellmax_bin = 2200
        ########
        # Cl_kgxg
        ########
        theory_kg = self.theory.get_Cl_kgxg()
        ell_theory_kg = theory_kg['ell']
        dl_1h_theory_kg = theory_kg['1h']
        dl_2h_theory_kg = theory_kg['2h']
        dl_gk_theory = np.asarray(list(dl_1h_theory_kg)) + np.asarray(list(dl_2h_theory_kg))
        #print('dl_gk_theory ', dl_gk_theory)
        ell_gk_binned, cl_gk_binned = self._bin(ell_theory_kg, dl_gk_theory, self.ell, ellmax_bin, bpwf, pixwin, Npoints, conv2cl=True)
        # print(cl_ell_theory)
        print('cl gk theory: ', dl_1h_theory_kg)
        print('cl gg: ', cl_gk_binned)
        # print('ell gg: ', ell_gg_binned)

        # ########
        # Cl_kgxmu
        ########
        theory_km = self.theory.get_Cl_kgxmu()
        ell_theory_km = theory_km['ell']
        dl_1h_theory_km = theory_km['1h']
        dl_2h_theory_km = theory_km['2h']
        dl_km_theory = np.asarray(list(dl_1h_theory_km)) + np.asarray(list(dl_2h_theory_km))
        ell_km_binned, cl_km_binned = self._bin(ell_theory_km, dl_km_theory, self.ell, ellmax_bin, bpwf, pixwin, Npoints, conv2cl=True)
        #print('cl gm: ', cl_gm_binned)

        # ########
        # Cl_IAxg
        ########
        theory_IA = self.theory.get_Cl_IAxg()
        ell_theory_IA = theory_IA['ell']
        dl_2h_theory_IA = theory_IA['2h']
        ell_IA_binned, cl_IA_binned = self._bin(ell_theory_km, dl_2h_theory_IA, self.ell, ellmax_bin,  bpwf, pixwin, Npoints, conv2cl=True)
        #print("cl_IA_2h: ", cl_IA_binned)

        f = ell_km_binned*(ell_km_binned+1)/2/np.pi
        cl_tot = cl_gk_binned + 2*(s-1)*cl_km_binned - cl_IA_binned
        print("ell bin: ", ell_km_binned[1:])
        print("total cl bin: ", cl_tot[1:])
        #print("total cl bin: ", (cl_tot*f)[1:])

        return cl_tot[1:]
