import torch

pthFile = r'/Users/akrik/Desktop/vkr/ids-backdoor/runs/Jul15_11-18-58_gpu_0_3/net_2175752832.pth'


# net = torch.load(pthFile, map_location=torch.device('cpu'))

# print(type(net))
# print(len(net))

# for k in net.keys():
#     print (k)

# for key,value in net["model"].items():
#     print(key,value.size(),sep="   ")

class EagerNet(torch.nn.Module):
	def __init__(self, n_input, n_output, n_layers, layer_size):
		device = "cpu"
		super(EagerNet, self).__init__()
		self.n_output = n_output
		self.n_layers = n_layers
		self.beginning = torch.nn.Linear(n_input, layer_size+n_output).to(device)
		self.middle = torch.nn.Sequential(*[torch.nn.Linear(layer_size, layer_size+n_output).to(device) for _ in range(n_layers)])
		self.end = torch.nn.Linear(layer_size, n_output).to(device)

	def forward(self, x):
		all_outputs = []
		all_xs = []
		output_beginning = self.beginning(x)
		all_xs.append(x)
		x = torch.nn.functional.leaky_relu(output_beginning[:,:-self.n_output])
		all_outputs.append(output_beginning[:,-self.n_output:])

		for current_layer in self.middle:
			current_output = current_layer(x)
			all_xs.append(x)
			x = torch.nn.functional.leaky_relu(current_output[:,:-self.n_output])
			all_outputs.append(current_output[:,-self.n_output:])

		all_xs.append(x)
		output_end = self.end(x)
		all_outputs.append(output_end)

		return all_outputs, all_xs


model = EagerNet(32, 15, 1, 128)
# model.load_state_dict(torch.load(pthFile, "cpu"))
# model.eval()

# output = model(1)


model = torch.load('runs/May07_21-48-37_MacBook-Air-Anton.local_0_3/net_m_744834.pth')

print(model)