from torch import nn


class TransformModule(nn.Module):
	def __init__(self, dim=(37, 60), num_view=1):
		super(TransformModule, self).__init__()
		self.num_view = num_view
		self.dim = dim
		self.mat_list = nn.ModuleList()

		for i in range(self.num_view):
			fc_transform = nn.Sequential(
						nn.Linear(dim[0] * dim[1], dim[0] * dim[1]),
						nn.ReLU(),
						nn.Linear(dim[0] * dim[1], dim[0] * dim[1]),
						nn.ReLU()
					)
			self.mat_list += [fc_transform]

	def forward(self, x):
		# shape x: B, V, C, H, W
		x = x.view(list(x.size()[:3]) + [self.dim[0] * self.dim[1],])
		view_comb = self.mat_list[0](x[:, 0])
		for index in range(x.size(1))[1:]:
			view_comb += self.mat_list[index](x[:, index])
		view_comb = view_comb.view(list(view_comb.size()[:2]) + list(self.dim))
		return view_comb

class Graph_Pred(nn.Module):
	def __init__(self, num_views, num_class, output_size, map_extents, map_resolution):

		super(Graph_Pred, self).__init__()
		self.num_views = num_views
		self.output_size = output_size

		self.seg_size = (
			int((map_extents[3] - map_extents[1]) / map_resolution),
			int((map_extents[2] - map_extents[0]) / map_resolution),
		)

		self.transform_module = TransformModule(dim=self.output_size, num_view=self.num_views)
		self.decoder = Graph(num_class)

	def forward(self, x, *args):
		B, N, C, H, W = x.view([-1, self.num_views, int(x.size()[1] / self.num_views)] \
							   + list(x.size()[2:])).size()

		x = x.view( B*N, C, H, W)
		x = x.view([B, N] + list(x.size()[1:]))
		x = self.transform_module(x)
		x = self.decoder(x)
		return x

class Graph(nn.Module):
	def __init__(self, num_class=3):
		super(Graph, self).__init__()
		self.pre_class = nn.Sequential(
			nn.AdaptiveAvgPool2d((16, 27)),
			# (16, 27) for normal; (16, 15) for sparse; (16, 21) for light; (16, 33) for thick; (16, 41) for dense
			nn.Conv2d(256, 512, kernel_size=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Dropout2d(0.1),
			nn.Conv2d(512, num_class, kernel_size=1)
			)
	def forward(self, x):
		x = self.pre_class(x)
		return x


