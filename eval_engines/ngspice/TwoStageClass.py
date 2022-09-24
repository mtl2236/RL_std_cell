import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
import yaml
import importlib
import time
import string
import math 

debug = False

#from eval_engines.ngspice.ngspice_wrapper import NgSpiceWrapper
from ngspice_wrapper import NgSpiceWrapper

class TwoStageClass(NgSpiceWrapper):
    Len_of_Index1=2
    Len_of_Index2=2

    
    
    def get_cell_succeed_percentage(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # # use parse output here
        # freq, vout,  ibias = self.parse_output(output_path)
        # gain = self.find_dc_gain(vout)
        # ugbw = self.find_ugbw(freq, vout)
        # phm = self.find_phm(freq, vout)


        # spec = dict(
            # ugbw=ugbw,
            # gain=gain,
            # phm=phm,
            # ibias=ibias
        # )
        cur_path=os.getcwd()
        os.chdir(output_path)
        f=open('char.log')
        log_list=f.readlines()
        f.close()
        flag0=0
        flag1=0
        success=0
        fail=0
        for i in range(len(log_list)):
            if ("Number of passing cells" in log_list[i]):
                success_str=log_list[i].split(' ')[-1]
                flag0=1
                success=float(success_str)
            if ("Number of failing cells" in log_list[i]):
                fail_str=log_list[i].split(' ')[-1]
                flag1=1
                fail=float(fail_str)
        if(flag0&flag1):
            spec=success/(success+fail)
        else:
            spec=0
        os.chdir(cur_path)    
        return spec
        
    def get_cell_pdp(self,output_path):
        cur_path=os.getcwd()
        os.chdir(output_path)
        #read the whole lines of the library file
        file_type_list = ['lib'] 
        filelist=self.get_file_list('lib',file_type_list)
        #print(filelist)
        f=open(filelist[0])
        Original_Lib=f.readlines()
        Cell_Begin_Str='cell '
        Cell_Begin_Line_Num=[]
        for i in range(len(Original_Lib)):
            if Cell_Begin_Str in Original_Lib[i]:
                Cell_Begin_Line_Num.append(i)
        Cell_Begin_Line_Num.append(len(Original_Lib)-1)
        #print(Cell_Begin_Line_Num)
        Cell_end_Line_Num=[]
        for j in range(len(Cell_Begin_Line_Num)-2):
            Cell_end_Line_Num.append(Cell_Begin_Line_Num[j+1]-1)
        Cell_end_Line_Num.append(Cell_Begin_Line_Num[j+2])
        #print(Cell_end_Line_Num)
        Cell_Begin_Line_Num.pop()  #delete the last line num
        #Extract cell name in every begin line
        # line_example=[]
        # line_example=Original_Lib[95].split(' ')
        # print(line_example)
        Cell_Name=[]
        for m in range(len(Cell_Begin_Line_Num)):
            str_split=[]
            str_split=Original_Lib[Cell_Begin_Line_Num[m]].split(' ')
            cell_name_str=str_split[3].strip('(')
            cell_name_str=cell_name_str.strip(')')
            Cell_Name.append(cell_name_str)
        #print(Cell_Name)
        #make pair for every cell begin line and end line
        Cell_Begin_End_Pair=[]
        for k in range(len(Cell_Begin_Line_Num)):
            Cell_Begin_End_Pair.append({'name':Cell_Name[k],'begin_line':Cell_Begin_Line_Num[k],'end_line':Cell_end_Line_Num[k],'original_PDP':0, 'M':1,'equivalent PDP':0})
        #print(Cell_Begin_End_Pair)
        #walk through every cell contents
        for n in range(len(Cell_Begin_End_Pair)):
            begin_line=Cell_Begin_End_Pair[n]['begin_line']
            end_line=Cell_Begin_End_Pair[n]['end_line']
            PDP_rise_index='rise_power (power_template)'
            PDP_fall_index='fall_power (power_template)'     #temporarily only for combinational cell
            PDP_rise_rawdata=[]
            PDP_fall_rawdata=[]
            #data processing ,some parameters should consider about Len_of_Index1 and Len_of_Index2
            for p in range(begin_line,end_line):
                if PDP_rise_index in Original_Lib[p]:
                    rise_rawdata0=[]
                    rise_rawdata1=[]
                    rise_rawdata0=Original_Lib[p+4].split(' ')
                    rise_rawdata0.pop()
                    del rise_rawdata0[0:12]
                    a=rise_rawdata0[0].strip('"')
                    a=a.strip(',')
                    num0=float(a)
                    b=rise_rawdata0[1].strip(',')
                    b=b.strip('"')
                    num1=float(b)
                    #print(num0)
                    #print(num1)
                    rise_rawdata1=Original_Lib[p+5].split(' ')
                    rise_rawdata1.pop()
                    del rise_rawdata1[0:12]
                    c=rise_rawdata1[0].strip('"')
                    c=c.strip(',')
                    num2=float(c)
                    d=rise_rawdata1[1].strip(',')
                    d=d.strip('"')
                    num3=float(d)
                    #print(num2)
                    #print(num3)
                    avg_rise_num=(num0+num1+num2+num3)/4
                    #print('avg!')
                    #print(avg_rise_num)
                    PDP_rise_rawdata.append(avg_rise_num)
                    
                    #print("rise!")
                    #print(avg_rise_num)
                
                if PDP_fall_index in Original_Lib[p]:
                    fall_rawdata0=[]
                    fall_rawdata1=[]
                    fall_rawdata0=Original_Lib[p+4].split(' ')
                    fall_rawdata0.pop()
                    del fall_rawdata0[0:12]
                    a=fall_rawdata0[0].strip('"')
                    a=a.strip(',')
                    num0=float(a)
                    b=fall_rawdata0[1].strip(',')
                    b=b.strip('"')
                    num1=float(b)
                    #print(num0)
                    #print(num1)
                    fall_rawdata1=Original_Lib[p+5].split(' ')
                    fall_rawdata1.pop()
                    del fall_rawdata1[0:12]
                    c=fall_rawdata1[0].strip('"')
                    c=c.strip(',')
                    num2=float(c)
                    d=fall_rawdata1[1].strip(',')
                    d=d.strip('"')
                    num3=float(d)
                    #print(num2)
                    #print(num3)
                    avg_fall_num=(num0+num1+num2+num3)/4
                    #print("fall!")
                    #print(avg_fall_num)
                    PDP_fall_rawdata.append(avg_fall_num)
            #delete repeating elements
            PDP_rise_rawdata=list(set(PDP_rise_rawdata))
            PDP_fall_rawdata=list(set(PDP_fall_rawdata))
            # print('rise!')
            # print(PDP_rise_rawdata)
            # print('fall!')
            # print(PDP_fall_rawdata)
            PDP_rise=np.mean(PDP_rise_rawdata)
            PDP_fall=np.mean(PDP_fall_rawdata)
            original_PDP=(PDP_rise+PDP_fall)/2
            Cell_Begin_End_Pair[n]['original_PDP']=original_PDP
            #print(original_PDP)
            #print(PDP_rise_rawdata)
            #print(PDP_fall_rawdata)
        #print(Cell_Begin_End_Pair)
        f.close()

        #reading M for each cell from external file

        f=open('M.txt','r+')
        M_file=f.readlines()
        strs=[]
        for i in range(len(M_file)):
            if(''!=M_file[i].split('\n')[0]):
                strs.append(M_file[i].split('\n')[0])
        #print(strs)
        names=[]
        M=[]
        for j in range(len(strs)):
            index=strs[j].find(' ')
            names.append(strs[j][0:index])
            M.append(strs[j][-1])
        #print(names)
        #print(M)
        M_vector=[]
        for k in range(len(names)):
            M_vector.append({'name':names[k],'M':M[k]})
        #print(M_vector) 
        Equivalent_PDP=0
        for u in range(len(M_vector)):
            for v in range(len(Cell_Begin_End_Pair)):
                if(M_vector[u]['name']==Cell_Begin_End_Pair[v]['name']):
                    Cell_Begin_End_Pair[v]['M']=M_vector[u]['M']
                    Cell_Begin_End_Pair[v]['equivalent PDP']=float(Cell_Begin_End_Pair[v]['original_PDP'])/float(Cell_Begin_End_Pair[v]['M'])
                    #print(Cell_Begin_End_Pair[v])
                    if(Cell_Begin_End_Pair[v]['equivalent PDP']>100):
                        Cell_Begin_End_Pair[v]['equivalent PDP']=10       #give this fail cell a big PDP but not infinite
                    Equivalent_PDP=Equivalent_PDP+Cell_Begin_End_Pair[v]['equivalent PDP']
        Equivalent_PDP=Equivalent_PDP/len(M_vector)
        #print(Equivalent_PDP)
        #normlaize_value=1/(1+math.exp(-Equivalent_PDP))    #logistic function 
        #print(normlaize_value)
        f.close()
        os.chdir(cur_path)
        return Equivalent_PDP
        

    # def parse_output(self, output_path):

        # ac_fname = os.path.join(output_path, 'ac.csv')
        # dc_fname = os.path.join(output_path, 'dc.csv')

        # if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            # print("ac/dc file doesn't exist: %s" % output_path)

        # ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        # dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        # freq = ac_raw_outputs[:, 0]
        # vout_real = ac_raw_outputs[:, 1]
        # vout_imag = ac_raw_outputs[:, 2]
        # vout = vout_real + 1j*vout_imag
        # ibias = -dc_raw_outputs[1]

        # return freq, vout, ibias

    # def find_dc_gain (self, vout):
        # return np.abs(vout)[0]

    # def find_ugbw(self, freq, vout):
        # gain = np.abs(vout)
        # ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        # if valid:
            # return ugbw
        # else:
            # return freq[0]

    # def find_phm(self, freq, vout):
        # gain = np.abs(vout)
        # phase = np.angle(vout, deg=False)
        # phase = np.unwrap(phase) # unwrap the discontinuity
        # phase = np.rad2deg(phase) # convert to degrees
        # #
        # # plt.subplot(211)
        # # plt.plot(np.log10(freq[:200]), 20*np.log10(gain[:200]))
        # # plt.subplot(212)
        # # plt.plot(np.log10(freq[:200]), phase)

        # phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        # ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        # if valid:
            # if phase_fun(ugbw) > 0:
                # return -180+phase_fun(ugbw)
            # else:
                # return 180 + phase_fun(ugbw)
        # else:
            # return -180


    # def _get_best_crossing(cls, xvec, yvec, val):
        # interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        # def fzero(x):
            # return interp_fun(x) - val

        # xstart, xstop = xvec[0], xvec[-1]
        # try:
            # return sciopt.brentq(fzero, xstart, xstop), True
        # except ValueError:
            # # avoid no solution
            # # if abs(fzero(xstart)) < abs(fzero(xstop)):
            # #     return xstart
            # return xstop, False

# class TwoStageMeasManager(object):

    # def __init__(self, design_specs_fname):
        # self.design_specs_fname = design_specs_fname
        # with open(design_specs_fname, 'r') as f:
            # self.ver_specs = yaml.load(f)

        # self.spec_range = self.ver_specs['spec_range']
        # self.params = self.ver_specs['params']

        # self.params_vec = {}
        # self.search_space_size = 1
        # for key, value in self.params.items():
            # if value is not None:
                # # self.params_vec contains keys of the main parameters and the corresponding search vector for each
                # self.params_vec[key] = np.arange(value[0], value[1], value[2]).tolist()
                # self.search_space_size = self.search_space_size * len(self.params_vec[key])

        # self.measurement_specs = self.ver_specs['measurement']
        # root_dir = self.measurement_specs['root_dir'] + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
        # num_process = self.measurement_specs['num_process']

        # self.netlist_module_dict = {}
        # for netlist_kwrd, netlist_val in self.measurement_specs['netlists'].items():
            # netlist_module = importlib.import_module(netlist_val['wrapper_module'])
            # netlist_cls = getattr(netlist_module, netlist_val['wrapper_class'])
            # self.netlist_module_dict[netlist_kwrd] = netlist_cls(num_process=num_process,
                                                                 # design_netlist=netlist_val['cir_path'],
                                                                 # root_dir=root_dir)

    # def evaluate(self, design):
        # state_dict = dict()
        # for i, key in enumerate(self.params_vec.keys()):
            # state_dict[key] = self.params_vec[key][design[i]]
        # state = [state_dict]
        # dsn_names = [design.id]
        # results = {}
        # for netlist_name, netlist_module in self.netlist_module_dict.items():
            # results[netlist_name] = netlist_module.run(state, dsn_names)

        # specs_dict = self._get_specs(results)
        # specs_dict['cost'] = self.cost_fun(specs_dict)
        # return specs_dict

    # def _get_specs(self, results_dict):
        # fdbck = self.measurement_specs['tb_params']['feedback_factor']
        # tot_err = self.measurement_specs['tb_params']['tot_err']

        # ugbw_cur = results_dict['ol'][0][1]['ugbw']
        # gain_cur = results_dict['ol'][0][1]['gain']
        # phm_cur = results_dict['ol'][0][1]['phm']
        # ibias_cur = results_dict['ol'][0][1]['Ibias']

        # # common mode gain and cmrr
        # cm_gain_cur = results_dict['cm'][0][1]['cm_gain']
        # cmrr_cur = 20 * np.log10(gain_cur / cm_gain_cur)  # in db
        # # power supply gain and psrr
        # ps_gain_cur = results_dict['ps'][0][1]['ps_gain']
        # psrr_cur = 20 * np.log10(gain_cur / ps_gain_cur)  # in db

        # # transient settling time and offset calculation
        # t = results_dict['tran'][0][1]['time']
        # vout = results_dict['tran'][0][1]['vout']
        # vin = results_dict['tran'][0][1]['vin']

        # tset_cur = self.netlist_module_dict['tran'].get_tset(t, vout, vin, fdbck, tot_err=tot_err)
        # offset_curr = abs(vout[0] - vin[0] / fdbck)

        # specs_dict = dict(
            # gain=gain_cur,
            # ugbw=ugbw_cur,
            # pm=phm_cur,
            # ibias=ibias_cur,
            # cmrr=cmrr_cur,
            # psrr=psrr_cur,
            # offset_sys=offset_curr,
            # tset=tset_cur,
        # )

        # return specs_dict

    # def compute_penalty(self, spec_nums, spec_kwrd):
        # if type(spec_nums) is not list:
            # spec_nums = [spec_nums]
        # penalties = []
        # for spec_num in spec_nums:
            # penalty = 0
            # spec_min, spec_max, w = self.spec_range[spec_kwrd]
            # if spec_max is not None:
                # if spec_num > spec_max:
                    # # penalty += w*abs((spec_num - spec_max) / (spec_num + spec_max))
                    # penalty += w * abs(spec_num - spec_max) / abs(spec_num)
            # if spec_min is not None:
                # if spec_num < spec_min:
                    # # penalty += w*abs((spec_num - spec_min) / (spec_num + spec_min))
                    # penalty += w * abs(spec_num - spec_min) / abs(spec_min)
            # penalties.append(penalty)
        # return penalties

    # def cost_fun(self, specs_dict):
        # """
        # :param design: a list containing relative indices according to yaml file
        # :param verbose:
        # :return:
        # """
        # cost = 0
        # for spec in self.spec_range.keys():
            # penalty = self.compute_penalty(specs_dict[spec], spec)[0]
            # cost += penalty

        # return cost
