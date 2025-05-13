import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class CurveData(Dataset):
    def __init__(self):
        #https://www.kaggle.com/datasets/talhabu/us-regional-sales-data
        df = pd.read_csv("US_Regional_Sales_Data.csv")

        #Order Quantity: Quantity of products ordered
        #Discount Applied: Applied discount for the order.
        #Unit Cost: Cost of a single unit of the product.
        #Unit Price: Price at which the product was sold.
        df = df[["Order Quantity", "Discount Applied", "Unit Cost", "Unit Price"]]
        df['Order Quantity'] = pd.to_numeric(df['Order Quantity'])
        df['Unit Cost'] = df['Unit Cost'].str.replace(',', '')
        df['Unit Price'] = df['Unit Price'].str.replace(',', '')
        df['Unit Cost'] = pd.to_numeric(df['Unit Cost'])
        df['Unit Price'] = pd.to_numeric(df['Unit Price'])
        # Feature and label data
        self.X = torch.tensor(df.iloc[:,-1], dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(df.iloc[:,-1], dtype=torch.float32)

        # Determine the length of the dataset
        self.len = self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X.view(-1)), np.array(self.y)


class CurveFit(nn.Module):
    def __init__(self):
        super(CurveFit, self).__init__()

        #Normalize
        self.norm = nn.BatchNorm1d(4)
        self.in_to_h1 = nn.Linear(1, 8)
        self.norm = nn.BatchNorm1d(4)
        self.h1_to_h2 = nn.Linear(8,4)
        self.h2_to_out = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x= F.relu(self.h1_to_h2(x))
        x = self.h2_to_out(x)
        return x


def trainNN(epochs=100, batch_size=16, lr=0.001, epoch_display=25):
    cd = CurveData()

    # Create a data loader that shuffles each time and allows for the last batch to be smaller
    # than the rest of the epoch if the batch size doesn't divide the training set size
    curve_loader = DataLoader(cd, batch_size=batch_size, drop_last=False, shuffle=True)

    # Create my neural network
    curve_network = CurveFit()

    # Mean square error (RSS)
    mse_loss = nn.MSELoss(reduction='sum')

    # Select the optimizer
    optimizer = torch.optim.Adam(curve_network.parameters(), lr=lr)

    running_loss = 0.0

    for epoch in range(epochs):
        for _, data in enumerate(curve_loader, 0):
            x, y = data

            optimizer.zero_grad()  # resets gradients to zero

            output = curve_network(x)  # evaluate the neural network on x

            loss = mse_loss(output.view(-1), y)  # compare to the actual label value

            loss.backward()  # perform back propagation

            optimizer.step()  # perform gradient descent with an Adam optimizer

            running_loss += loss.item()  # update the total loss

        # every epoch_display epochs give the mean square error since the last update
        # this is averaged over multiple epochs
        if epoch % epoch_display == epoch_display - 1:
            print(
                f"Epoch {epoch + 1} / {epochs} Average loss: {running_loss / (len(cd) * epoch_display):.6f}")
            running_loss = 0.0
    return curve_network, cd


cn, cd = trainNN(epochs=50)

with torch.no_grad():
    y_pred = cn(cd.X).view(-1)
    y_linear = cn(torch.linspace(0, 5, steps=100).view(-1,1))

X_numpy, y_numpy = cd.to_numpy()

print(f"MSE (fully trained): {np.average((y_numpy-np.array(y_pred))**2)}")

plt.scatter(X_numpy, y_pred, s=10)
plt.plot(np.linspace(0, 5, num=100), np.array(y_linear), c='r')
plt.show()