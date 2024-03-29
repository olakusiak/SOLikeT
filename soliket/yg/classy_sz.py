from cobaya.theories.classy import classy
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional
from cobaya.tools import load_module
import logging
import os

class classy_sz(classy):
    def initialize(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""
        self.classy_module = self.is_installed()
        if not self.classy_module:
            raise NotInstalledError(
                self.log, "Could not find CLASS_SZ. Check error message above.")
        from classy_sz import Class, CosmoSevereError, CosmoComputationError
        global CosmoComputationError, CosmoSevereError
        self.classy = Class()
        super(classy,self).initialize()
        # Add general CLASS stuff
        self.extra_args["output"] = self.extra_args.get("output", "")
        if "sBBN file" in self.extra_args:
            self.extra_args["sBBN file"] = (
                self.extra_args["sBBN file"].format(classy=self.path))
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []
        self.log.info("Initialized!")

        # class_sz default params for lkl
        # self.extra_args["output"] = 'tSZ_1h'
        # self.extra_args["multipoles_sz"] = 'P15'
        # self.extra_args['nlSZ'] = 18



    # def must_provide(self, **requirements):
    #     if "Cl_sz" in requirements:
    #         # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("Cl_sz")
    #         # specify the method to collect the new observable
    #         self.collectors["Cl_sz"] = Collector(
    #                 method="cl_sz", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     if "Cl_yxg" in requirements:
    #         # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("Cl_yxg")
    #         # specify the method to collect the new observable
    #         self.collectors["Cl_yxg"] = Collector(
    #                 method="cl_yg", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     if "Cl_gxg" in requirements:
    #         # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("Cl_gxg")
    #         # specify the method to collect the new observable
    #         self.collectors["Cl_gxg"] = Collector(
    #                 method="cl_gg", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     if "Cl_gxmu" in requirements:
    #         # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("Cl_gxmu")
    #         # specify the method to collect the new observable
    #         self.collectors["Cl_gxmu"] = Collector(
    #                 method="cl_gm", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     if "Cl_muxmu" in requirements:
    #         # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("Cl_muxmu")
    #         # specify the method to collect the new observable
    #         self.collectors["Cl_muxmu"] = Collector(
    #                 method="cl_mm", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     if "Cl_kxg" in requirements:
    #         # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("Cl_kxg")
    #         # specify the method to collect the new observable
    #         self.collectors["Cl_kxg"] = Collector(
    #                 method="cl_kg", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     if "Cl_kxmu" in requirements:
    #     # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("Cl_kxmu")
    #     # specify the method to collect the new observable
    #         self.collectors["Cl_kxmu"] = Collector(
    #                 method="cl_km", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     if "Cl_yxmu" in requirements:
    #         # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("Cl_yxmu")
    #         # specify the method to collect the new observable
    #         self.collectors["Cl_yxmu"] = Collector(
    #                 method="cl_ym", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     if "s8omegamp5" in requirements:
    #         # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("s8omegamp5")
    #         # specify the method to collect the new observable
    #         self.collectors["s8omegamp5"] = Collector(
    #                 method="s8omegamp5", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     if "H0" in requirements:
    #         # make sure cobaya still runs as it does for standard classy
    #         requirements.pop("H0")
    #         # specify the method to collect the new observable
    #         self.collectors["H0"] = Collector(
    #                 method="H0", # name of the method in classy.pyx
    #                 args_names=[],
    #                 args=[])
    #     super().must_provide(**requirements)

    # get the required new observable
    def get_Cl_sz(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_sz"])
        return cls

    def get_Cl_yxg(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_yxg"])
        return cls
    def get_Cl_kxg(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_kxg"])
        return cls
    def get_Cl_kgxg(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_kgxg"])
        return cls
    def get_Cl_gxg(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_gxg"])
        return cls
    def get_Cl_muxmu(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_muxmu"])
        return cls
    def get_Cl_gxmu(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_gxmu"])
        return cls
    def get_Cl_kxmu(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_kxmu"])
        return cls
    def get_Cl_yxmu(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_yxmu"])
        return cls
    def get_Cl_galnxlens(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_galnxlens"])
        return cls
    def get_Cl_galnxgaln(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_galnxgaln"])
        return cls
    def get_Cl_galnxtsz(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_galnxtsz"])
        return cls
    def get_Cl_galnxgallens(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_galnxgallens"])
        return cls
    def get_Cl_lensmagnxtsz(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_lensmagnxtsz"])
        return cls
    def get_Cl_lensmagnxgallens(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_lensmagnxgallens"])
        return cls
    def get_Cl_galn_IA(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_galnxIA"])
        return cls

    @classmethod
    def is_installed(cls, **kwargs):
        return load_module('classy_sz')

# this just need to be there as it's used to fill-in self.collectors in must_provide:
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
