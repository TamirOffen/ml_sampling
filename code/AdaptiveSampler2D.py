import numpy as np

class AdaptiveSampler2D:
    """
    Adaptive sampler that is specific to a 2 arm link robot
    """
    def __init__(self, legal_config_func, resolution=0.025):
        self.resolution = resolution
        self.angle_range = np.arange(0, 2 * np.pi, resolution)
        self.pdf = self.init_pdf()
        self.legal_config_func = legal_config_func
        self.X_obs = set()
        self.X_free = set()

    def init_pdf(self):
        num_of_cells = len(self.angle_range)
        pdf = np.ones((num_of_cells, num_of_cells)) / num_of_cells**2
        return pdf

    def sample(self):
        pdf_flat = self.pdf.flatten()
        sampled_index = np.random.choice(self.pdf.size, p=pdf_flat)
        x, y = np.unravel_index(sampled_index, self.pdf.shape)
        
        theta_1 = x * self.resolution
        theta_2 = y * self.resolution
        return (theta_1, theta_2)

    def density_estimator(self, X, x_sample):
        """
        calculates P( x_sample | y=... ) according to X (obs or free)
        """
        b = 0
        for x in X:
            h_f = (np.log(len(X)+1) / len(X)) ** 0.5
            x_new = (x_sample[0] - x[0], x_sample[1] - x[1])
            b += self.multi_var_kernel(x_new, h_f, "epanechnikov")
        return 0 if len(X) == 0 else  b / len(X)

    def update_sampling_pdf(self, x):

        approx_X_free = set()
        for theta_1 in self.angle_range:
            for theta_2 in self.angle_range:
                x = (theta_1, theta_2)
                pred = self.classifier(x)
                if pred == 1: 
                    approx_X_free.add(x)

        # print(f'X : {(len(self.angle_range))**2}')
        # print(f'approx X free {len(approx_X_free)}')

        self.pdf.fill(0)
        for theta_1, theta_2 in approx_X_free:
            a, b = int(theta_1 / self.resolution), int(theta_2 / self.resolution)
            self.pdf[a][b] = 1

        # normalize the pdf to ensure the sum of probabilities is 1
        self.pdf /= np.sum(self.pdf)


    def classifier(self, x):
        """
        x is in the config space (2 angles)
        return 1 if q_free(x) >= q_obs(x), else -1
        """
        if len(self.X_free) + len(self.X_obs) == 0: # no samples
            return -1 
        
        theta_1, theta_2 = x
        a, b = int(theta_1 / self.resolution), int(theta_2 / self.resolution)

        if self.pdf[a][b] < 1e-12:  # Check if pdf is close to zero
            return -1  # or any other handling approach
            
        eta = 1 / self.pdf[a][b] # 1 / P(x)
        b_obs = self.density_estimator(self.X_obs, x) # P(x | y=-1)
        b_free = self.density_estimator(self.X_free, x) # P(x | y=1)
        p_obs = len(self.X_obs) / (len(self.X_free) + len(self.X_obs)) # P(y=-1)
        p_free = len(self.X_free) / (len(self.X_free) + len(self.X_obs)) # P(y=1)

        q_free = eta * b_free * p_free
        q_obs = eta * b_obs * p_obs
        return 1 if q_free >= q_obs else -1

    def run(self, num_iterations, print_every=0):
        for i in range(num_iterations):
            if print_every > 0:
                if i % print_every == 0: print(f'iter {i}')

            x_rand = self.sample() #(theta1, theta2)
            if self.legal_config_func(x_rand):
                self.X_free.add(x_rand)
            else: # only update the pdf if col
                self.X_obs.add(x_rand)
                self.update_sampling_pdf(x_rand)
        X = (self.X_obs, self.X_free)
        return X
    
    def kernel(self, x, type):
        """
        type = {uniform, gaussian, epanechnikov}
        """
        if type == "uniform":
            return (1/(2**2)) if np.dot(x,x) <=1 else 0
        if type == "gaussian":
            exp = np.exp(-0.5 * np.dot(x, x))
            return exp / ((2 * np.pi)**(2/2))
        if type == "epanechnikov":
            # with formula:
            # mu = np.pi * 1**2
            # return ((2+2)*(1-np.dot(x,x))) / (2 * mu) if np.dot(x,x) <= 1 else 0
            
            # efficient way:
            inner_prod = x[0]**2 + x[1]**2
            return 0 if inner_prod > 1 else (2-2*inner_prod)/np.pi
        
        print("warning: kernel type not valid")
        return 0
    
    def multi_var_kernel(self, x, h, type):
        # general formula, but very inefficient:
        # H = np.diag([h]*2)
        # det = np.linalg.det(H)
        # H_inv = np.linalg.inv(H)
        # return self.kernel((H_inv @ x), type) / det

        # using lin alg tricks for diag matrix:
        x_1, x_2 = x
        return self.kernel((x_1/h, x_2/h), type) / (h**2)

