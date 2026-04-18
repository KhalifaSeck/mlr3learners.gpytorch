import torch
import gpytorch
import numpy as np


class ExactGPModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, kernel="rbf"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchWrapper:
    
    def __init__(self, kernel="rbf", lr=0.1, n_iter=50):
        self.kernel = kernel
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.model = None
        self.likelihood = None
        self.train_x = None
        self.train_y = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
    
    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std == 0] = 1.0
        X_norm = (X - self.x_mean) / self.x_std
        
        self.y_mean = float(y.mean())
        self.y_std = float(y.std())
        if self.y_std == 0:
            self.y_std = 1.0
        y_norm = (y - self.y_mean) / self.y_std
        
        self.train_x = torch.from_numpy(X_norm).float()
        self.train_y = torch.from_numpy(y_norm).float()
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(
            self.train_x, self.train_y, self.likelihood, self.kernel
        )
        
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for i in range(self.n_iter):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
        
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        
        X_norm = (X - self.x_mean) / self.x_std
        test_x = torch.from_numpy(X_norm).float()
        
        self.model.eval()
        self.likelihood.eval()
       
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(test_x))
            mean_norm = pred.mean.numpy()
            var_norm = pred.variance.numpy()
    
        mean = mean_norm * self.y_std + self.y_mean
        variance = var_norm * (self.y_std ** 2)
        
        return {'mean': mean, 'variance': variance}