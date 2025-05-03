data_path = "../data/"
model_path = '../model/'
result_path = "../result/"
fig_path = "../fig/"
file = "train_data_with_mantle_types_with_continental_arc_basalts.xlsx"

major_element = ['SiO2', 'TiO2', 'Al2O3', 'CaO', 'MgO', 'MnO', 'K2O', 'Na2O']
# , 'P2O5'
# trace_element = ['Sc', 'V', 'Cr', 'Ni', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
trace_element = ["Ni", "Sr", "Y", "Zr", "Nb", "Ba", "La", "Ce"]

top_features = ['TA(PPM)', 'HF(PPM)', 'ND(PPM)', 'PB(PPM)', 'SR(PPM)', 'SC(PPM)', 'LA(PPM)', 'GD(PPM)', 'EU(PPM)', 'CE(PPM)', 'U(PPM)', 'NB(PPM)', 'BA(PPM)', 'TIO2(WT%)', 'TB(PPM)', 'YB(PPM)', 'LU(PPM)', 'K2O(WT%)', 'TH(PPM)']
top_features = ['TA(PPM)', 'HF(PPM)', 'ND(PPM)', 'PB(PPM)', 'SC(PPM)', 'SR(PPM)', 'EU(PPM)', 'LA(PPM)', 'GD(PPM)', 'TIO2(WT%)', 'U(PPM)', 'BA(PPM)', 'CE(PPM)', 'NB(PPM)', 'YB(PPM)', 'SM(PPM)', 'TB(PPM)', 'TH(PPM)', 'NI(PPM)']
#op_features = ['HF(PPM)', 'SR(PPM)', 'TA(PPM)', 'GD(PPM)', 'ND(PPM)', 'CR(PPM)', 'TIO2(WT%)', 'Y(PPM)', 'PB(PPM)', 'ZR(PPM)', 'CE(PPM)', 'SM(PPM)', 'NA2O(WT%)', 'P2O5(WT%)', 'SC(PPM)', 'TH(PPM)', 'U(PPM)', 'NB(PPM)', 'EU(PPM)', 'LA(PPM)']
# "Ti",不在这里面;'FeOT', 这个在预测集中没有过
major_element_upper = [s.upper() + "(WT%)" for s in major_element]
trace_element_upper = [s.upper() + "(PPM)" for s in trace_element]
elements = major_element + trace_element
elements_upper = major_element_upper + trace_element_upper


