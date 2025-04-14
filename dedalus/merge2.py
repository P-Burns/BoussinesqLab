
import subprocess
import os
from dedalus.tools import post
import sys


if len(sys.argv) > 1:
    RunName = str(sys.argv[1])
    singlePoint = int(sys.argv[2])
    linear = int(sys.argv[3])

print(RunName)

if float(RunName) < 10: RunName = '0' + RunName
RunName = RunName.replace('.', '_')
if singlePoint: RunName = RunName + '_sp'
if linear: RunName = RunName + '_lnr'
#post.merge_process_files('Results/StateN2_' + str(RunName), cleanup=True)
post.merge_process_files('Results_bigNu/StateN2_' + str(RunName), cleanup=True)
    
