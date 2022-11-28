import numpy as np
from zmqRemoteApi import RemoteAPIClient
import matplotlib.pyplot as plt
import time

np.set_printoptions(suppress=True, precision=4)


def c(theta):
    return np.cos(theta)


def s(theta):
    return np.sin(theta)


def t0_1(theta):
    return np.array([
        [c(theta), 0, -s(theta), 0],
        [s(theta), 0, c(theta), 0],
        [0, -1, 0, 0.11],
        [0, 0, 0, 1]
    ])


def t1_2(theta):
    return np.array([
        [c((2 * theta - np.pi) / 2), -s((2 * theta - np.pi) / 2), 0, 0.125 * c((2 * theta - np.pi) / 2)],
        [s((2 * theta - np.pi) / 2), c((2 * theta - np.pi) / 2), 0, 0.125 * s((2 * theta - np.pi) / 2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def t2_3(theta):
    return np.array([
        [c((2 * theta + np.pi) / 2), -s((2 * theta + np.pi) / 2), 0, 0.096 * c((2 * theta + np.pi) / 2)],
        [s((2 * theta + np.pi) / 2), c((2 * theta + np.pi) / 2), 0, 0.096 * s((2 * theta + np.pi) / 2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def t3_4(theta):
    return np.array([
        [c((2 * theta + np.pi) / 2), 0, s((2 * theta + np.pi) / 2), -0.0275 * c((2 * theta + np.pi) / 2)],
        [s((2 * theta + np.pi) / 2), 0, -c((2 * theta + np.pi) / 2), -0.0275 * s((2 * theta + np.pi) / 2)],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])


def t4_5(theta):
    return np.array([
        [c(theta), -s(theta), 0, 0],
        [s(theta), c(theta), 0, 0],
        [0, 0, 1, 0.065],
        [0, 0, 0, 1]
    ])


def fkine(q):
    return t0_1(q[0]) @ t1_2(q[1]) @ t2_3(q[2]) @ t3_4(q[3]) @ t4_5(q[4])


def quat2euler(h):
    roll = np.arctan2(2 * (h[0] * h[1] + h[2] * h[3]), 1 - 2 * (h[1] ** 2 + h[2] ** 2))
    pitch = np.arcsin(2 * (h[0] * h[2] - h[3] * h[1]))
    yaw = np.arctan2(2 * (h[0] * h[3] + h[1] * h[2]), 1 - 2 * (h[2] ** 2 + h[3] ** 2))

    return (roll, pitch, yaw)


def matrix_union(A, B):
    for a, b in zip(A, B):
        yield [*a, *b]


def jacobian(q):
    t01 = t0_1(q[0])
    t12 = t1_2(q[1])
    t23 = t2_3(q[2])
    t34 = t3_4(q[3])
    t45 = t4_5(q[4])

    rotation0_1 = t01[:3, :3]
    rotation1_2 = t12[:3, :3]
    rotation2_3 = t23[:3, :3]
    rotation3_4 = t34[:3, :3]
    rotation4_5 = t45[:3, :3]

    p0 = np.array([0, 0, 0]).reshape(3, 1)

    p1 = t01[0:3, -1].reshape(3, 1)

    p2 = (t01 @ t12)[0:3, -1].reshape(3, 1)

    p3 = (t01 @ t12 @ t23)[0:3, -1].reshape(3, 1)

    p4 = (t01 @ t12 @ t23 @ t34)[0:3, -1].reshape(3, 1)

    p5 = (t01 @ t12 @ t23 @ t34 @ t45)[0:3, -1].reshape(3, 1)

    z0 = np.array([0, 0, 1]).reshape(3, 1)

    z1 = np.dot(rotation0_1, z0)

    z2 = np.dot(rotation0_1, rotation1_2)
    z2 = np.dot(z2, z0)

    z3 = np.dot(rotation0_1, rotation1_2)
    z3 = np.dot(z3, rotation2_3)
    z3 = np.dot(z3, z0)

    z4 = np.dot(rotation0_1, rotation1_2)
    z4 = np.dot(z4, rotation2_3)
    z4 = np.dot(z4, rotation3_4)
    z4 = np.dot(z4, z0)

    x0 = np.cross(z0.T, (p5 - p0).T).T
    x1 = np.cross(z1.T, (p5 - p1).T).T
    x2 = np.cross(z2.T, (p5 - p2).T).T
    x3 = np.cross(z3.T, (p5 - p3).T).T
    x4 = np.cross(z4.T, (p5 - p4).T).T

    conc1 = np.concatenate((x0, x1), axis=1)
    conc2 = np.concatenate((x2, x3), axis=1)
    conc3 = np.concatenate((conc2, x4), axis=1)

    up_part = np.concatenate((conc1, conc3), axis=1)

    conc_under1 = np.concatenate((z0, z1), axis=1)
    conc_under2 = np.concatenate((z2, z3), axis=1)
    conc_under3 = np.concatenate((conc_under2, z4), axis=1)

    down_part = np.concatenate((conc_under1, conc_under3), axis=1)

    return np.concatenate((up_part, down_part))


print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

dummy_handle = sim.getObject('/Dummy')

position = sim.getObjectPosition(dummy_handle, -1)
orientation = sim.getObjectQuaternion(dummy_handle, -1)

quat = np.array([orientation[3], orientation[0], orientation[1],
                 orientation[2]])  # Remember: getObjectQuaternion has real part as last element

(alpha, beta, gamma) = sim.getObjectOrientation(dummy_handle, -1)
(yaw, pitch, roll) = sim.alphaBetaGammaToYawPitchRoll(alpha, beta, gamma)

X_d = np.array([[position[0]], [position[1]], [position[2]], [roll], [pitch], [yaw]])

j1 = sim.getObject('/theta1')
j2 = sim.getObject('/theta2')
j3 = sim.getObject('/theta3')
j4 = sim.getObject('/theta4')
j5 = sim.getObject('/theta5')

theta1 = sim.getJointPosition(j1)
theta2 = sim.getJointPosition(j2)
theta3 = sim.getJointPosition(j3)
theta4 = sim.getJointPosition(j4)
d4 = sim.getJointPosition(j5)

q = np.array([theta1, theta2, theta3, theta4, d4])

init_q = q
T = fkine(q)

R = T[0:3, 0:3]

for i in range(3):
    for j in range(3):
        if (np.abs(R[i][j]) < 0.01):
            R[i][j] = 0.0

alpha = np.arctan2(R[1][0], R[0][0])
beta = np.arctan2(-R[2][0], np.sqrt((R[2][1] ** 2) + (R[2][2] ** 2)))
gamma = np.arctan2(R[2][1], R[2][2])

X_m = np.vstack([T[0:3, -1].reshape(3, 1), np.array([alpha, beta, gamma]).reshape((3, 1))])

time_left = 0

manipulability = []
effector_x = []
effector_y = []
effector_z = []
theta_1 = []
theta_2 = []
theta_3 = []
theta_4 = []
theta_5 = []

effector = sim.getObject('/end_effector_visual')

Ts = 0.1

Tempo = np.arange(0, 15, Ts)

while (time_left < 15):
    error = X_d - X_m
    J = jacobian(q)

    try:
        manipulability.append(np.sqrt(np.linalg.det(J @ J.T)))
    except:
        manipulability.append(0)

    (x, y, z) = sim.getObjectPosition(effector, -1)

    effector_x.append(x)
    effector_y.append(y)
    effector_z.append(z)

    theta_1.append(q[0])
    theta_2.append(q[1])
    theta_3.append(q[2])
    theta_4.append(q[3])
    theta_5.append(q[4])

    J_pinv = (np.transpose(J) @ np.linalg.inv(J @ np.transpose(J) + 0.5 ** 2 * np.eye(6)))  # Pseudoinversa amortecida
    dq = J_pinv @ error  # Cinemática diferencial
    dq = dq.reshape((5,))

    q = q + dq * Ts

    sim.setJointTargetPosition(j1, q[0])
    sim.setJointTargetPosition(j2, q[1])
    sim.setJointTargetPosition(j3, q[2])
    sim.setJointTargetPosition(j4, q[3])
    sim.setJointTargetPosition(j5, q[4])

    time.sleep(Ts)

    theta1 = sim.getJointPosition(j1)
    theta2 = sim.getJointPosition(j2)
    theta3 = sim.getJointPosition(j3)
    theta4 = sim.getJointPosition(j4)
    d4 = sim.getJointPosition(j5)

    q = np.array([theta1, theta2, theta3, theta4, d4])

    T = fkine(q)

    R = T[0:3, 0:3]

    for i in range(0, 3):
        for j in range(0, 3):
            if (np.abs(R[i][j]) < 0.1):
                R[i][j] = 0.0

    alpha = np.arctan2(R[1][0], R[0][0])

    beta = np.arctan2(-R[2][0], np.sqrt((R[2][1] ** 2) + (R[2][2] ** 2)))

    gamma = np.arctan2(R[2][1], R[2][2])

    X_m = np.vstack([T[0:3, -1].reshape(3, 1), np.array([alpha, beta, gamma]).reshape((3, 1))])

    position = sim.getObjectPosition(dummy_handle, -1)
    orientation = sim.getObjectQuaternion(dummy_handle, -1)

    quat = np.array([orientation[3], orientation[0], orientation[1],
                     orientation[2]])  # Remember: getObjectQuaternion has real part as last element

    (alpha, beta, gamma) = sim.getObjectOrientation(dummy_handle, -1)
    (yaw, pitch, roll) = sim.alphaBetaGammaToYawPitchRoll(alpha, beta, gamma)

    X_d = np.array([[position[0]], [position[1]], [position[2]], [roll], [pitch], [yaw]])

    time_left = time_left + Ts


print("Matriz jacobiana no início:\n", jacobian(init_q))
print()
print("Matriz jacobiana ao final:\n", jacobian(q))

plt.figure()
plt.subplot(311)
print(Tempo.shape, len(manipulability))
plt.scatter(Tempo, manipulability[:Tempo.shape[0]], s=1.5, c='blue')
plt.title('Manipulabilidade do sistema')
plt.grid(0.5)
plt.ylabel('Manipulabilidade')

plt.subplot(312)
plt.plot(Tempo, effector_x[:Tempo.shape[0]], 'r', label='Posição X do efetuador')
plt.plot(Tempo, effector_y[:Tempo.shape[0]], 'black', label='Posição Y do efetuador')
plt.plot(Tempo, effector_z[:Tempo.shape[0]], 'blue', label='Posição Z do efetuador')
plt.legend(loc='best', framealpha=1)
plt.title('Posição das coordenadas')
plt.grid(0.5)
plt.ylabel('Coordenadas')

plt.subplot(313)
plt.plot(Tempo, theta_1[:Tempo.shape[0]], 'g', label='Theta 1')
plt.plot(Tempo, theta_2[:Tempo.shape[0]], 'r', label='Theta 2')
plt.plot(Tempo, theta_3[:Tempo.shape[0]], 'y', label='Theta 3')
plt.plot(Tempo, theta_4[:Tempo.shape[0]], 'b', label='Theta 4')
plt.plot(Tempo, theta_5[:Tempo.shape[0]], 'k', label='Theta 5')
plt.legend(loc='best', framealpha=1)
plt.title('Posição das Juntas')
plt.grid(0.5)
plt.xlabel('Tempo(s)')
plt.ylabel('Juntas')
plt.show()
