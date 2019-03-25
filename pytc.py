import lightspeed as ls
import psiw
from base_lot import Lot 
import numpy as np
import manage_xyz
from units import *
from collections import Counter
from rhf_lot import RHF_LOT
from casci_lot_svd import CASCI_LOT_SVD
import sys
from nifty import custom_redirection

class PyTC(Lot):
    """
    Level of theory is a wrapper object to do DFT and CASCI calculations 
    Inherits from Lot. Requires a PSIW object
    """

    def get_energy(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        return self.getE(state,multiplicity)

    def getE(self,state,multiplicity):
        tmp = self.search_tuple(self.E,multiplicity)
        return tmp[state][1]*KCAL_MOL_PER_AU

    def get_mm_energy(self,coords):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            self.psiw.update_qmmm(coords*ANGSTROM_TO_AU)
        if self.psiw.__class__.__name__=="CASCI_LOT" or self.psiw.__class__.__name__=="CASCI_LOT_SVD":
            return self.psiw.casci.ref.geometry.qmmm.mm_energy
        else:
            return self.psiw.rhf.geometry.qmmm.mm_energy

    def get_mm_gradient(self,coords):
        #TODO need diff variable for hasRan MM energy
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            self.psiw.update_qmmm(coords*ANGSTROM_TO_AU)
        if self.psiw.__class__.__name__=="CASCI_LOT" or self.psiw.__class__.__name__=="CASCI_LOT_SVD":
            return self.psiw.casci.ref.geometry.qmmm.mm_gradient
        else:
            return self.psiw.rhf.geometry.qmmm.mm_gradient

    def run(self,geom,verbose=False):
        self.E=[]
        self.grada=[]
        #normal update
        coords = manage_xyz.xyz_to_np(geom)
        T = ls.Tensor.array(coords*ANGSTROM_TO_AU)
        def run_code(T):
            self.psiw = self.psiw.update_xyz(T)
            for state in self.states:
                multiplicity=state[0]
                ad_idx=state[1]
                S=multiplicity-1
                if self.psiw.__class__.__name__=="CASCI_LOT" or self.psiw.__class__.__name__=="CASCI_LOT_SVD":
                    self.E.append((multiplicity,self.psiw.compute_energy(S=S,index=ad_idx)))
                    tmp = self.psiw.compute_gradient(S=S,index=ad_idx)
                elif self.psiw.__class__.__name__=="RHF_LOT": 
                    self.E.append((multiplicity,self.psiw.compute_energy()))
                    tmp = self.psiw.compute_gradient()
                self.grada.append((multiplicity,tmp[...]))
            if self.do_coupling==True:
                state1=self.states[0][1]
                state2=self.states[1][1]
                tmp = self.psiw.compute_coupling(S=S,indexA=state1,indexB=state2)
                self.coup = tmp[...]

        if not verbose:
            with open('psiw_jobs.txt','a') as out:
                with custom_redirection(out):
                    run_code(T)
        else:
            run_code(T)

                #filename="{}_rhf_update.molden".format(self.node_id)
                #self.psiw.casci.reference.save_molden_file(filename)

        self.hasRanForCurrentCoords=True
        return

    def getgrad(self,state,multiplicity):
        tmp = self.search_tuple(self.grada,multiplicity)
        return np.asarray(tmp[state][1])*ANGSTROM_TO_AU

    def get_gradient(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        return self.getgrad(state,multiplicity)

    def getcoup(self,state1,state2,multiplicity):
        #TODO this could be better
        return np.reshape(self.coup,(3*len(self.coup),1))*ANGSTROM_TO_AU

    def get_coupling(self,coords,multiplicity,state1,state2):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        return self.getcoup(state1,state2,multiplicity)

    def copy(self,node_id):
        """ create a copy of this psiw object"""
        do_coupling = self.do_coupling
        obj = PyTC(self.options.copy().set_values({
            "node_id" :node_id,
            "do_coupling":do_coupling,
            }))

        #if self.psiw.__class__.__name__=="CASCI_LOT":
        #    obj.psiw = psiw.CASCI_LOT(self.psiw.options.copy())
        #            
        #elif self.psiw.__class__.__name__=="RHF_LOT": 
        #    obj.psiw = RHF_LOT(self.psiw.options.copy())
        obj.psiw = type(obj.psiw)(self.psiw.options.copy())
                    
        return obj

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return PyTC(PyTC.default_options().set_values(kwargs))


