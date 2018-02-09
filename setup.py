import os, sys
import fileinput
from subprocess import Popen, PIPE

accepted = set(["y", "Y", "YES", "yes" ])
denied = set(["n", "N", "NO", "no" ])

testedDistributions = ["14.04", "15.04"]

def ask( question ):
  accept = None
  while accept not in accepted.union(denied):
    accept = raw_input( question )
  if accept in accepted: return True
  if accept in denied: return False

def replaceAll(file,searchExp,replaceExp):
  for line in fileinput.input(file, inplace=1):
    if searchExp in line:
      line = line.replace(searchExp,replaceExp)
    sys.stdout.write(line)

#Get Ubuntu Distribution
(stdout, stderr) = Popen(["lsb_release","-a"], stdout=PIPE).communicate()
stdout = stdout.split()
ubuDist = stdout[stdout.index("Release:") + 1 ]
print "Ubuntu Distribution: " + ubuDist
if ubuDist not in testedDistributions:
  if not ask("This installer has not been tested in this Ubuntu Distribution. Continue?(y/n) "): exit("Goodbye")


ready = False
homeDir = os.path.expanduser('~')
currentDir = os.getcwd()
line = "\nEnter the directory path were phyGPU enviroment will be added\nDefault: {0} \nHit ENTER to use default current directory\n".format(currentDir)
while not ready:
  ans = raw_input(line)
  if ans == '': envDir = currentDir
  else: envDir = ans
  question = 'pyhGPU will be installed in: {0}  .Is this OK?(y/n)'.format(envDir)
  ready = ask( question )
if not os.path.exists( envDir ): os.makedirs( envDir )

# line_toFind = 'ENV_DIR=$HOME_DIR'
# line_toReplace = 'ENV_DIR="{0}"'.format(envDir)
# replaceAll( 'install.sh', line_toFind, line_toReplace )

#Install the virtual environment
# os.system('bash install.sh {0}'.format(envDir))
