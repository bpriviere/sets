
import torch
import numpy as np 
import time as timer
from tqdm import tqdm 
from torch.nn.utils.parametrizations import spectral_norm

import plotter


# torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.double)

class Feedforward(torch.nn.Module):

    def __init__(self, 
            device,
            overfit_mode,
            training_mode,
            lipshitz_const,
            train_test_ratio,
            batch_size,
            initial_learning_rate,
            num_hidden_layers,
            num_epochs,
            model_path,
            input_dim,
            hidden_dim,
            output_dim):
        super(Feedforward, self).__init__()

        self.device = device
        self.overfit_mode = overfit_mode
        self.training_mode = training_mode
        self.lipshitz_const = lipshitz_const
        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size
        self.initial_learning_rate = initial_learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.activation = torch.relu
        # self.activation = torch.nn.Tanh()
        # self.activation = torch.nn.Linear()

        # param 
        # if self.spectral_norm_on:
        self.layers = torch.nn.ModuleList()
        if self.training_mode == "spectral_normalization":
            self.layers.append(spectral_norm(torch.nn.Linear(self.input_dim, self.hidden_dim)))
            self.layers.extend([spectral_norm(torch.nn.Linear(self.hidden_dim, self.hidden_dim)) for _ in range(self.num_hidden_layers)])
            self.layers.append(spectral_norm(torch.nn.Linear(self.hidden_dim, self.output_dim)))
        else:
            self.layers.append(torch.nn.Linear(self.input_dim, self.hidden_dim))
            self.layers.extend([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_hidden_layers)])
            self.layers.append(torch.nn.Linear(self.hidden_dim, self.output_dim))
        self.to("cpu")


    def extract_ff_layers(self):
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        keys.sort()
        weightss, biass = [], []
        # if self.spectral_norm_on: 
        if self.training_mode == "spectral_normalization": 
            for key in keys: 
                if "bias" in key: 
                    biass.append(state_dict[key].numpy())
                if "weight.original" in key: 
                    weightss.append((self.lipshitz_const ** (1 / (len(weightss)+1))) * state_dict[key].numpy())
        else:
            for key in keys: 
                if "bias" in key: 
                    biass.append(state_dict[key].numpy())
                if "weight" in key: 
                    weightss.append(state_dict[key].numpy())
        return weightss, biass

    def __call__(self, x):
        # x is torch tensor in (bs, n)
        # y is torch tensor in (bs, m) 
        # y = x
        # for k, layer in enumerate(self.layers[:-1]):
        #     # if self.spectral_norm_on:
        #     if self.training_mode == "spectral_normalization":
        #         y = self.activation((self.lipshitz_const ** (1 / (k+1))) * layer(y))
        #     else:
        #         y = self.activation(layer(y))
        # y = self.layers[-1](y)

        y = x
        for k, layer in enumerate(self.layers[:-1]):
            y = self.activation(layer(y))
        y = self.layers[-1](y)
        return y

    def np_call(self, x):
        # x is np array in (n,) 
        torch_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # (1, n)
        torch_y = self(torch_x) # (1, output)
        y = torch_y.detach().numpy()[0,:].astype(np.double) # in (m,)
        return y

    def train(self, X_np, Y_np):

        X_np = X_np.astype(np.float32)
        Y_np = Y_np.astype(np.float32)

        # make loaders
        if self.overfit_mode:
            train_dataset = Dataset(X_np, Y_np, device=self.device)
            test_dataset = Dataset(X_np, Y_np, device=self.device)
        else:
            split_idx = int(X_np.shape[0] * self.train_test_ratio)
            train_dataset = Dataset(X_np[0:split_idx,:], Y_np[0:split_idx,:], device=self.device)
            test_dataset = Dataset(X_np[split_idx:,:], Y_np[split_idx:,:], device=self.device)
        self.batch_size = min((self.batch_size, len(test_dataset)))
        print("self.batch_size",self.batch_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', \
            factor=0.5, patience=20, min_lr=1e-4*self.initial_learning_rate, verbose=True)

        # send everything to same device 
        train_dataset.to(self.device)
        test_dataset.to(self.device)
        self.to(self.device)

        # # train! 
        losses = []
        best_test_loss = np.Inf
        pbar = tqdm(range(self.num_epochs))
        start_time = timer.time()
        epochs_since_last_update = 0
        curr_learning_rate = optimizer.param_groups[0]["lr"]
        for epoch in range(self.num_epochs): 
            train_epoch_loss = train(self, optimizer, train_loader)
            test_epoch_loss = test(self, test_loader)
            scheduler.step(test_epoch_loss)
            losses.append((train_epoch_loss, test_epoch_loss))
            epochs_since_last_update += 1
            if curr_learning_rate != optimizer.param_groups[0]["lr"]:
                self.load(self.model_path)
                curr_learning_rate = optimizer.param_groups[0]["lr"]
            if test_epoch_loss < best_test_loss:
                epochs_since_last_update = 0
                best_test_loss = test_epoch_loss
                self.to("cpu")
                self.save(self.model_path)
                self.to(self.device)
            pbar.set_description("epoch: {}, train_epoch_loss: {:.7f}, test_epoch_loss: {:.7f}, best_test_loss: {:.7f}".format(
                epoch, train_epoch_loss, test_epoch_loss, best_test_loss))
            pbar.update(1)
            if epochs_since_last_update > 500:
                print("500 epochs since last best test loss, breaking training loop")
                break 
        self.to("cpu")
        print("training time: {}".format(timer.time() - start_time))

        # plotter.plot_histogram(X_np, ["px", "py", "pz", "vx", "vy", "vz", "roll", 
        #                               "pitch", "yaw", "rollRate", "pitchRate", "yawRate", "t", 
        #                               "thrust_z", "tau_x", "tau_y", "tau_z"])
        # plotter.plot_histogram(Y_np, ["fx", "fy", "fz", "mx", "my", "mz"])

        # plotting 
        fig, ax = plotter.make_fig()
        losses_np = np.array(losses) # nepochs x 2
        ax.plot(losses_np[:,0], color="blue", label="Train")
        ax.plot(losses_np[:,1], color="orange", label="Test")
        ax.legend()
        ax.set_yscale("log")

        return None

    # https://www.google.com/search?channel=fs&client=ubuntu&q=lipshitz+regularization+training#fpstate=ive&vld=cid:19214e28,vid:fRfGaqiTe-o
    def compute_lipshitz_constant(self):
        constant = 1.0 
        for layer in self.layers:
            constant *= torch.norm(layer.weight)
        return constant


    def load(self, path): 
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)


    def loss(self, y_model, y_data):
        loss_func = torch.nn.MSELoss()
        # mse = loss_func(y_model, y_data)
        mse = torch.mean(torch.norm(y_model - y_data, dim=1) / torch.clamp(torch.norm(y_data, dim=1), min=1e-5)) 
        if self.training_mode == "lipshitz_regularization":
            # print("yes")
            # mse += self.lipshitz_const * torch.norm(y_data) * self.compute_lipshitz_constant()
            mse += self.lipshitz_const * self.compute_lipshitz_constant()
        return mse


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, X_np, Y_np, device='cpu'):
        self.X_torch = torch.tensor(X_np, device=device)
        self.Y_torch = torch.tensor(Y_np, device=device)
        self.to(device)

    def __len__(self):
        return self.X_torch.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        return self.X_torch[idx,:], self.Y_torch[idx,:]

    def to(self,device):
        self.X_torch = self.X_torch.to(device)
        self.Y_torch = self.Y_torch.to(device)


def train(model, optimizer, loader):
    epoch_loss = 0
    for step, (x, y_data) in enumerate(loader):
        y_model = model(x)
        loss = model.loss(y_model, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss)
    return epoch_loss/len(loader)


def test(model, loader):
    epoch_loss = 0
    for step, (x, y_data) in enumerate(loader):
        y_model = model(x)
        loss = model.loss(y_model, y_data)
        epoch_loss += float(loss)
    return epoch_loss/len(loader)
