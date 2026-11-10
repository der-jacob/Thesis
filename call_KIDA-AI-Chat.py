# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:23:22 2026

@author: opitz
"""


import os
from openai import OpenAI
import pandas as pd
import time
#from  sklearn.metrics import cohen_kappa_score, accuracy_score, \
#   confusion_matrix, ConfusionMatrixDisplay, f1_score

system_prompt_file = 'Epi-Science_system_prompt.txt'
screening_file     = 'Data_All_OhneExtraCC.csv'
output_file        = 'Validation_SD_3.csv'

#---BEGIN: set up objects for LLM use------------------------------------------
# set no proxy so we can reach the server
try:
    os.environ['NO_PROXY'] = os.environ['NO_PROXY'] + ',' + 'https://openwebui.kida.bfr.bund.de'
except:
    os.environ['NO_PROXY'] = 'https://openwebui.kida.bfr.bund.de'
    
# API configuration
api_key  = 'sk-f10eae64d26746629a22487a7650a1aa'
base_url = 'https://openwebui.kida.bfr.bund.de/openai/'

# Setup OpenAI client
client = OpenAI(
    api_key  = api_key,
    base_url = base_url
)

model = "openai/gpt-oss-120b" # Choose any available model
#---END: set up objects for LLM use--------------------------------------------

#---BEGIN: Prepare input objects, like sys prompt and data---------------------
# Read file with system prompt
path_to_sys_prompt = "Sys-Prompts/" + system_prompt_file
with open(path_to_sys_prompt, mode = "r",  encoding = 'utf-8') as f:
    sys_prompt = f.read()

# read csv with validation data
path_to_val_data = "Screening_data/" + screening_file
df_screen = pd.read_csv(path_to_val_data, sep = ";", index_col = False)
df_screen["Title"] = df_screen["Title"].fillna("")
df_screen["Abstract"] = df_screen["Abstract"].fillna("")

# create clean  list with observed values
# obs = [this.lower().strip() for this in df_screen.Consensus_annotation]
# print(set(obs))
# for study_type in set(obs):
#     print(study_type, "=", obs.count(study_type))
#---END: Prepare input objects, like sys prompt and data---------------------

#---BEGIN: Classify Paper using title and abstract-----------------------------
# how many observations are there
n = len(df_screen.index)

# list for predicted labels
pred = []

# prediciton loop
start = time.time()
for i in range(n):
    # set up prompt for classifcation
    prompt = 'To which class belongs the following text: ' + \
        df_screen["Title"][i] + ". " + df_screen["Abstract"][i]

    #Get response
    chat_completion = client.chat.completions.create(
         messages         = [{"role"    : "system",
                              "content" : sys_prompt},
                             {"role"    : "user",
                              "content" : prompt}],
         model            = model,
         temperature      = 0.0,
         reasoning_effort = "medium"
    )
  
    tmp = chat_completion.choices[0].message.content
    
    pred.append(tmp)
    
    print(i+1, ".", tmp)
    
end = time.time()
  
print("Time for", model, ":", round((end - start)/60, 2), "minutes")
print("Time per paper for", model, ":", round((end - start)/n, 2), "sec\n")
#---END: Classify Paper suing title and abstract-------------------------------

#---BEGIN: clean predicted labels----------------------------------------------
print(set(pred))
pred_clean = []
for i in range(n):
    
    # remove extra text
#    if pred[i] is not None:
#        tmp = pred[i].split("\n")
#else:
 #   tmp = []
    tmp = pred[i].split("\n")
    
    if len(tmp) > 1:
        tmp = tmp[1]
        
    pred_clean.append(tmp[0])
print(set(pred_clean))
#---END: clean predicted labels------------------------------------------------

#---BEGIN: add predictions as new column in dataframe, and saving it-----------
df_screen['pred'] = pred_clean

path_to_output_file = 'Results/' + output_file
df_screen.to_csv(path_to_output_file, index = False)

#---END: add predictions as new column in dataframe, and saving it-------------

#---BEGIN: get statistics------------------------------------------------------
# kappa = cohen_kappa_score(obs, pred_clean)
# acc = accuracy_score(obs, pred_clean)
# f1 = f1_score(obs, pred_clean, average = "weighted")
# print("kappa       =", round(kappa, 3))
# print("acc         =", round(acc, 3))
# print("f1 weighted =", round(f1, 3))
# cm = confusion_matrix(obs, pred_clean, labels = list(set(obs)))
# print(cm)
# cm_disp = ConfusionMatrixDisplay(cm, 
#                                 display_labels = list(set(obs)))
#cm_disp.plot()
#---END: get statistics--------------------------------------------------------
