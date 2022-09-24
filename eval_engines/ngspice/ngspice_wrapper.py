import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint
import yaml
import shutil
#import IPython
debug = False

class NgSpiceWrapper(object):

    BASE_TMP_DIR = os.path.abspath("/simulation/tianliang/sim_da")

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        if root_dir == None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        # with open(yaml_path, 'r') as f:
            # yaml_data = yaml.full_load(f)
        # design_netlist = yaml_data['dsn_netlist']
        # design_netlist = path+'/'+design_netlist
 
        # _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = 'std_cell_sim'
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_liberate")

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        # raw_file = open(design_netlist, 'r')
        # self.tmp_lines = raw_file.readlines()
        # raw_file.close()

    # def get_design_name(self, state):
        # fname = self.base_design_name
        # for value in state.values():
            # fname += "_" + str(value)
        # return fname
    def get_file_list(self, folder, file_type_list):
        filelist = []  
        for dirpath,dirnames,filenames in os.walk(folder):
            for file in filenames:
                file_type = file.split('.')[-1]
                if(file_type in file_type_list):
                    file_fullname = os.path.join(dirpath, file) #full name of files
                    filelist.append(file_fullname)
        return filelist
    
    
    def create_design(self, state):
        #design_folder = os.path.join(self.gen_dir, new_fname)+str(random.randint(0,10000))
        design_folder = self.gen_dir+str(random.randint(0,100000))
        os.makedirs(design_folder, exist_ok=True)
        #copy workspace into this folder
        #shutil.copytree("/simulation/tianliang/test/RL_std_cell/eval_engines/ngspice/ngspice_inputs/liberate", design_folder+"/liberate")
        os.system('cp -r /simulation/tianliang/test/RL_std_cell/eval_engines/ngspice/ngspice_inputs/liberate '+design_folder+'/liberate')
        fpath = design_folder+'/liberate'

        #modify netlist from state
        cur_path=os.getcwd()
        os.chdir(fpath)
        file_type_list = ['scs'] 
        filelist=self.get_file_list('netlist', file_type_list)
        for l in range(len(filelist)):
            f=open(filelist[l],'r')
            #print(filelist[l])
            netlist_file=f.readlines()
            f.close()
            #print(netlist_file)
            #update parameters
            for m in range(len(netlist_file)):
                #update parameters
                if('(' in netlist_file[m]):
                    #print(netlist_file[m])
                    strs=netlist_file[m].split(' ')
                    strs[7]='MU='+str(state['MU'])
                    strs[8]='COX='+str(state['COX'])
                    if('-' in strs[13]):
                        strs[13]='VTO=-'+str(state['VTO'])
                    else:
                        strs[13]='VTO='+str(state['VTO'])
                    strs[21]='VSS='+str(state['VSS'])
                    strs[28]='RCS='+str(state['RCS'])
                    strs[29]='RCD='+str(state['RCD'])
                    #write back to .scs file
                    new_line=''
                    for u in range(len(strs)):
                        new_line=new_line+strs[u]+' '
                    new_line=new_line.strip()
                    new_line=new_line+'\n'
                    #print(new_line)
                    netlist_file[m]=new_line
            f=open(filelist[l],'w')
            for m in range(len(netlist_file)):
                f.writelines(netlist_file[m])
            f.close()    
        
        #update VDD
        f=open('tcl/char.tcl','r')
        char_tcl=f.readlines()
        f.close() 
        char_tcl[8]='set VDD           '+str(state['VDD'])+'\n'
        f=open('tcl/char.tcl','w')
        for i in range(len(char_tcl)):
            f.writelines(char_tcl[i])
        f.close()
        os.chdir(cur_path)
        
        #print(VDD)
        #print(VSS)
        
        # lines = copy.deepcopy(self.tmp_lines)
        # for line_num, line in enumerate(lines):
            # if '.include' in line:
                # regex = re.compile("\.include\s*\"(.*?)\"")
                # found = regex.search(line)
                # if found:
                    # # current_fpath = os.path.realpath(__file__)
                    # # parent_path = os.path.abspath(os.path.join(current_fpath, os.pardir))
                    # # parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
                    # # path_to_model = os.path.join(parent_path, 'spice_models/45nm_bulk.txt')
                    # # lines[line_num] = lines[line_num].replace(found.group(1), path_to_model)
                    # pass # do not change the model path
            # if '.param' in line:
                # for key, value in state.items():
                    # regex = re.compile("%s=(\S+)" % (key))
                    # found = regex.search(line)
                    # if found:
                        # new_replacement = "%s=%s" % (key, str(value))
                        # lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)
            # if 'wrdata' in line:
                # regex = re.compile("wrdata\s*(\w+\.\w+)\s*")
                # found = regex.search(line)
                # if found:
                    # replacement = os.path.join(design_folder, found.group(1))
                    # lines[line_num] = lines[line_num].replace(found.group(1), replacement)

        # with open(fpath, 'w') as f:
            # f.writelines(lines)
            # f.close()
        return design_folder, fpath

    def simulate(self, fpath):
        #info = 0 # this means no error occurred
        cur_path=os.getcwd()
        os.chdir(fpath)
        command = "liberate tcl/char.tcl 2>&1|tee char.log"
        os.system(command)
        os.chdir(cur_path)
        # exit_code = os.system(command)
        if debug:
            print(command)
            print(fpath)

        # if (exit_code % 256):
           # # raise RuntimeError('program {} failed!'.format(command))
            # info = 1 # this means an error has occurred
        # return info


    def create_design_and_simulate(self, state):
        # if debug:
            # print('state', state)
            # print('verbose', verbose)
        # if dsn_name == None:
            # dsn_name = self.get_design_name(state)
        # else:
            # dsn_name = str(dsn_name)
        # if verbose:
            # print(dsn_name)
        design_folder, fpath = self.create_design(state)
        self.simulate(fpath)
        # specs = self.translate_result(design_folder)
        succeed_rate=self.get_cell_succeed_percentage(fpath)
        pdp=self.get_cell_pdp(fpath)
        #r=random.random()
        #return state, specs, info
        return (float(succeed_rate*100)-float(pdp))


    def run(self, states, design_names=None, verbose=False):
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate, arg_list)
        pool.close()
        return specs

    def translate_result(self, output_path):
        """
        This method needs to be overwritten according to cicuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.

        :param output_path:
        :return:
        """
        result = None
        return result
