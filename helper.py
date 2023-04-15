import csv
import os
import random

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels

class Dataset():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.transform = transform
        self.data = []
        self.allowed__columns = """Crash_ID Crash_Fatal_Fl	Schl_Bus_Fl	Rr_Relat_Fl	Medical_Advisory_Fl	
        Active_School_Zone_Fl  Rpt_Outside_City_Limit_Fl	Thousand_Damage_Fl	Rpt_Latitude	Rpt_Longitude	
        Private_Dr_Fl	Toll_Road_Fl	Crash_Speed_Limit	Road_Constr_Zone_Fl	Road_Constr_Zone_Wrkr_Fl Wthr_Cond_ID	Light_Cond_ID	
        Road_Type_ID	Road_Algn_ID	Surf_Cond_ID	Traffic_Cntl_ID         Cnty_ID	City_ID	Latitude	Longitude	
        Txdot_Rptable_Fl	Onsys_Fl	Rural_Fl	Crash_Sev_ID	Pop_Group_ID	Located_Fl         
        Hp_Shldr_Left	Hp_Shldr_Right	Hp_Median_Width Nbr_Of_Lane	Row_Width_Usual	Roadbed_Width	Surf_Width	Surf_Type_ID	Curb_Type_Left_ID	
        Curb_Type_Right_ID	Shldr_Type_Left_ID	Shldr_Width_Left	Shldr_Use_Left_ID	Shldr_Type_Right_ID	Shldr_Width_Right	Shldr_Use_Right_ID	
        Median_Type_ID	Median_Width	Rural_Urban_Type_ID	Func_Sys_ID	Adt_Curnt_Amt	Adt_Curnt_Year	Adt_Adj_Curnt_Amt	Pct_Single_Trk_Adt	
        Pct_Combo_Trk_Adt	Trk_Aadt_Pct Sus_Serious_Injry_Cnt	Nonincap_Injry_Cnt	Poss_Injry_Cnt	Non_Injry_Cnt	Unkn_Injry_Cnt	Tot_Injry_Cnt	
        Death_Cnt"""

        #Hwy_Nbr Day_of_Week Hwy_Sys Rpt_Rdwy_Sys_ID	Rpt_Hwy_Num	Rpt_Hwy_Sfx	Rpt_Road_Part_ID	Rpt_Block_Num	Rpt_Street_Pfx	Rpt_Street_Name	Rpt_Street_Sfx
        self.allowed__columns = self.allowed__columns.split()

        for dirs in os.listdir('data/'):
            print(dirs)
            yearly_data = None
            for file in os.listdir('data/'+ dirs):
                temp = pd.read_csv(os.path.join('data/'+ dirs +"/"+ file))
                if yearly_data is not None:
                    yearly_data = yearly_data.merge(temp, on='Crash_ID', how = 'outer')
                else:
                    yearly_data = temp
            self.data.append(yearly_data)

        self.data = pd.concat([self.data[0], self.data[1], self.data[2], self.data[3], self.data[4]], axis=0)

        self.data = self.data[self.allowed__columns]
        self.labels = torch.tensor(self.data['Crash_Sev_ID'].values)
        self.data.drop(['Crash_Sev_ID'], axis = 1, inplace = True)

        self.data = self.data.fillna(0)
        self.data = self.data.replace('N', 0, regex=True)
        self.data = self.data.replace('Y', 1, regex=True)

        self.data = self.data.values
        print(self.data.shape)

        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2,
                                                            stratify=self.labels)

        if mode =="train":
            self.data, self.labels = x_train, y_train
        else:
            self.data, self.labels = x_test, y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        safety = self.data[idx]
        safety = safety.reshape(1, self.size, 1)
        label = int(self.labels[idx])
        #label = random.randint(0,3)
        sample = {'safety': safety, 'label': label}

        return sample