# Make sure to have the server side running in CoppeliaSim:
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

try:
    import sim
except:
    print('--------------------------------------------------------------')
    print('"sim.py" could not be imported. This means very probably that')
    print('either "sim.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "sim.py"')
    print('--------------------------------------------------------------')
    print('')

import time
import math
import sys
import numpy as np

from zmqRemoteApi import RemoteAPIClient

print('Program started')
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

client = RemoteAPIClient()
sim_ = client.getObject('sim')

if clientID != -1:
    print('Connected to remote API server\n')
    time.sleep(0.02)
    err, dummy = sim.simxGetObjectHandle(clientID, 'Dummy', sim.simx_opmode_oneshot_wait)
    err, floor = sim.simxGetObjectHandle(clientID, 'Floor', sim.simx_opmode_oneshot_wait)

    # Criando stream de dados
    err, position_dummy = sim.simxGetObjectPosition(clientID, dummy, floor, sim.simx_opmode_streaming)
    err, orientation_dummy = sim.simxGetObjectOrientation(clientID, dummy, floor, sim.simx_opmode_streaming)

    time.sleep(5)

    err, position_dummy = sim.simxGetObjectPosition(clientID, dummy, floor, sim.simx_opmode_buffer)
    err, orientation_dummy = sim.simxGetObjectOrientation(clientID, dummy, floor, sim.simx_opmode_buffer)

    orientation_dummy_degree = [d * (180 / math.pi) for d in orientation_dummy]
    print(f"Position Dummy: {position_dummy}\n")
    print(f"Orientation Dummy: {orientation_dummy_degree}\n")

    matrix_transformation = sim_.getObjectMatrix(dummy, floor)
    matrix_transformation_formatted = np.array(matrix_transformation).reshape((3, 4))

    print(f"Transformation Matrix: \n{matrix_transformation_formatted}\n")

    sim.simxGetPingTime(clientID)

    sim.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')