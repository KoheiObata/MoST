import numpy as np
import matplotlib.pyplot as plt

seed=0
np.random.seed(seed)

def genInvCov(size, low = 0.3 , upper = 0.6, portion = 0.2,symmetric = True):
    portion = portion/2
    A=np.zeros([size,size])
    if size==1:
        return A
    for i in range(size):
        for j in range(size):
            if i>=j:
                continue
            coin=np.random.uniform()
            if coin<portion:
                value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
                A[i,j] = value
    if np.allclose(A,np.zeros([size,size])):
        i,j=0,0
        while i==j:
            i=np.random.randint(0,size,1)
            j=np.random.randint(0,size,1)
        value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
        A[i,j] = value
    if symmetric:
        A = A + A.T
    return np.matrix(A)

def genInvCov_rectangle(size1, size2, low = 0.3 , upper = 0.6, portion = 0.2):
    portion = portion/2
    A=np.zeros([size1,size2])
    for i in range(size1):
        for j in range(size2):
            # if i>=j:
                # continue
            coin=np.random.uniform()
            if coin<portion:
                value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
                A[i,j] = value
    if np.allclose(A,np.zeros([size1,size2])):
        i,j=0,0
        while i==j:
            i=np.random.randint(0,size1,1)
            j=np.random.randint(0,size2,1)
        value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
        A[i,j] = value
    return np.matrix(A)

def nonlinear_trend(T, beta0=0.2, beta1=60):
	y = np.empty(T)
	for t in range(T):
		y[t] = 1 / (1 + np.exp(beta0*(t - beta1))) + np.random.normal(loc=0, scale=0.3)
	return y

def linear_trend(T):
	alpha = np.random.uniform(0.3,1)
	if np.random.uniform(-1,1) < 0:
		alpha *= -1
	y = np.empty(T)
	for t in range(T):
		y[t] = alpha * t/T
	return y

def sin_seasonal(T, cycles=3):
	# cycles: how many sine cycles
	# T: how many datapoints to generate

	length = np.pi * 2 * cycles
	my_wave = np.sin(np.arange(0, length, length / T))
	return my_wave

def get_latent_d(T, d, cycle_st=1, cycle_ed=20):
	latent_d = np.empty(shape=(T, d))
	for i in range(d):
		y = linear_trend(T)
		y += sin_seasonal(T, cycles=np.random.randint(cycle_st,cycle_ed))
		# y += np.random.normal(loc=0, scale=0.5, size=T)
		latent_d[:, i] = y
	return latent_d

def get_multi_latent_d(latent_d):
	d = latent_d.shape[-1]
	A = genInvCov(d, portion=0.3)
	A += np.eye(d)
	multi_latent_d = np.dot(latent_d, A)
	return multi_latent_d, A

def get_multi_latent_d_rectangle(latent_d, d):
	ld = latent_d.shape[-1]
	A = genInvCov_rectangle(ld, d, portion=0.3)
	multi_latent_d = np.dot(latent_d, A)
	return multi_latent_d, A

def add_multi_latent_d1_d2(multi_latent_d1, multi_latent_d2):
	T, d1 = multi_latent_d1.shape
	T, d2 = multi_latent_d2.shape
	data = np.empty(shape=(T, d1, d2))
	for i in range(d1):
		for j in range(d2):
			data[:, i, j] = np.squeeze(multi_latent_d1[:, i] + multi_latent_d2[:, j])
			data[:, i, j] += np.random.normal(loc=0, scale=0.3, size=T)
	return data


# change only A
T = 1000
d1 = 20
d2 = 20

latent_d1 = 320
latent_d2 = 320

n_d1 = 3
n_d2 = 3

latent_d1 = get_latent_d(T, latent_d1)
latent_d2 = get_latent_d(T, latent_d2)

print('latent_d1',latent_d1.shape)
print('latent_d2',latent_d2.shape)


multi_latent_d1_list = []
multi_latent_d2_list = []
A_d1_list = []
A_d2_list = []
for n1 in range(n_d1):
	multi_latent_d1, A_d1 = get_multi_latent_d_rectangle(latent_d1, d1)
	multi_latent_d1_list.append(multi_latent_d1)
	A_d1_list.append(A_d1)
for n2 in range(n_d2):
	multi_latent_d2, A_d2 = get_multi_latent_d_rectangle(latent_d2, d2)
	multi_latent_d2_list.append(multi_latent_d2)
	A_d2_list.append(A_d2)

data_list = []
label_list = []
for n1 in range(n_d1):
	for n2 in range(n_d2):
		data = add_multi_latent_d1_d2(multi_latent_d1_list[n1], multi_latent_d2_list[n2])
		data_list.append(data)
		label_list.append([n1,n2])

print([_.shape for _ in data_list])
print(label_list)
data = np.stack(data_list, axis=0)
np.save(f'./synthetic/synthetic_{d1}_{d2}_{n_d1}_{n_d2}.npy',data)
