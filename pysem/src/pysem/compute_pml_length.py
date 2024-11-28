# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script to compute PML properties for SEM3D

    Ex.1 : Compute amplitude Ax knowning PML length (300. m):
        
        python3 compute_pml_length.py @@PML_length 300.
    
    Ex.2 : Compute PML_length knowing the amplitude Ax (10.)
        
        python3 compute_pml_length.py @@Ax 10.
"""
# Required modules
import sys
import argparse
import numpy as np

# General informations
__author__ = "Filippo Gatti"
__copyright__ = "Copyright 2020, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"


class pml(object):
    def __init__(self,**kwargs):
        self.__call__(**kwargs)
        
    def __call__(self,**kwargs):
        self.__dict__.update(**kwargs)
        self.setup()
        self.check()
    
    def setup(self):
        assert len(self.cp)==len(self.cs)
        self.cp,self.cs = np.array(self.cp),np.array(self.cs)
        self.PML_lengths = np.array(self.PML_lengths)
        self.Lor = self.cs/self.fl[0]                                                       
        self.ll=(self.cs/self.fl[1],self.cp/self.fl[0])
        self.kl=(2.*np.pi/self.ll[1],2.*np.pi/self.ll[0])
        self.RC = 10.**(self.RCdb/20.)
        print(f"cp: {self.cp} m/s")
        print(f"cs: {self.cs} m/s")
        print(f"Frequency limits: {self.fl} Hz")
        print(f"wave-length limits: {self.ll} m")
        print(f"wave-number limits: {self.kl} 1/m")
        print(f"Reflection coefficient: {self.RC}")

    def check(self):
        self.flag = []
        if self.Ax is None:
            self.flag.append('amp')
        if None in self.PML_lengths:
            self.flag.append('len')
        if 'amp' in self.flag:
            if None not in self.PML_lengths:
                self.get_amplitude()
            else:
                raise ValueError('PML lengths not defined!')
        if 'len' in self.flag:
            if self.Ax is not None:
                self.get_length()
            else:
                raise ValueError('PML amplitude not defined!')
            
    def get_length(self):
        if self.PML_type == 'PML':
            self.PML_lengths = (-0.5*((self.p+1)*np.log(self.RC)/self.kl[0]/self.Ax))**(1./(self.p+1))
        elif self.PML_type == 'CPML':
            self.PML_lengths = -0.5/self.Ax*(self.p+1)*self.cp*np.log(self.RC)
        print("PML lengths ({}): {}".format(self.PML_type,self.PML_lengths))
    
    def get_amplitude(self):
        if self.PML_type == 'PML':
            self.Ax = -0.5*(self.p+1)*np.log(self.RC)/(self.kl[0]*self.PML_lengths**(self.p+1))                      
        elif self.PML_type== 'CPML':
            self.Ax = -0.5/self.PML_lengths*(self.p+1)*self.cp*np.log(self.RC)
        print(f"Ax ({self.PML_type}) = {self.Ax}")

def get_options():
    parser = argparse.ArgumentParser(prefix_chars='@' )
    parser.add_argument('@@cp',type=float,nargs='+',default=[ 700.,1385.,1732.,3500.],help="P-wave speed in each layer")
    parser.add_argument('@@cs',type=float,nargs='+',default=[ 300., 800.,1000.,2000.],help="S-wave speed in each layer")
    parser.add_argument('@@fl',type=float,nargs='+',default=[0.01,30.],help="Frequency limits")
    parser.add_argument('@@RCdb',type=float,default=-60.,help="Reflection Coefficient in db")
    parser.add_argument('@p',type=int,default=2,help="Polynomial order")
    parser.add_argument('@@Ax',type=float,default=None,help="Ax or d0 coefficient")
    parser.add_argument('@@PML_lengths',type=float,nargs='+',default=None,help="PML length")
    parser.add_argument('@@PML_type',type=str,default="PML",help="PML type [PML|CPML]")
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    opt = parser.parse_args().__dict__
    
    return opt

def main(opt: dict|None = None):
    if opt is None:
        opt = get_options()
    p = pml(opt)

if __name__=="__main__":
    main()
