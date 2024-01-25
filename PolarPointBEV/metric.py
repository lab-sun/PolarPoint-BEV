from sklearn.metrics import f1_score, classification_report
import torch
import numpy as np
import json


def List2List(List):
	Arr1 = np.array(List[:-1]).reshape(-1, List[0].shape[1])
	Arr2 = np.array(List[-1]).reshape(-1, List[0].shape[1])
	Arr = np.vstack((Arr1, Arr2))

	return [i for item in Arr for i in item]

def Gragh_Metric(total_gt_graph, total_pred_graph):
	graph_num = len(total_gt_graph)
	gt_graph_list = []
	pred_graph_list = []

	for idx in range(graph_num):
		gt_graph = total_gt_graph[idx].cpu().numpy()
		pred_graph = torch.argmax(total_pred_graph[idx], dim=1)
		pred_graph = pred_graph.cpu().numpy()
		gt_graph_list.append(gt_graph)
		pred_graph_list.append(pred_graph)


	gt_graph_list = List2List(gt_graph_list)
	pred_graph_list = List2List(pred_graph_list)

	# Overall F1 for graph
	f1_graph_overall = f1_score(gt_graph_list, pred_graph_list, average='micro')
	# Mean F1 for graph
	f1_graph_mean = f1_score(gt_graph_list, pred_graph_list, average='macro')

	return f1_graph_overall, classification_report(gt_graph_list, pred_graph_list), f1_graph_mean


def Cal_IoU(total_gt_bev, total_pred_bev):
	confmat = ConfusionMatrix(3)
	for idx in range(len(total_gt_bev)):
		confmat.update(total_gt_bev[idx].flatten(), total_pred_bev[idx].argmax(1).flatten())
	confmat.reduce_from_all_processes()

	return confmat

class ConfusionMatrix(object):
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.mat = None

	def update(self, a, b):
		n = self.num_classes
		if self.mat is None:
			# 创建混淆矩阵
			self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
		with torch.no_grad():
			# 寻找GT中为目标的像素索引
			k = (a >= 0) & (a < n)
			# 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
			inds = n * a[k].to(torch.int64) + b[k]
			self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

	def reset(self):
		if self.mat is not None:
			self.mat.zero_()

	def compute(self):
		h = self.mat.float()
		acc_global = torch.diag(h).sum() / h.sum()
		acc = torch.diag(h) / h.sum(1)
		iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
		return acc_global, acc, iu

	def reduce_from_all_processes(self):
		if not torch.distributed.is_available():
			return
		if not torch.distributed.is_initialized():
			return
		torch.distributed.barrier()
		torch.distributed.all_reduce(self.mat)

	def __str__(self):
		acc_global, acc, iu = self.compute()
		return (
			'global correct: {:.1f}\n'
			'average row correct: {}\n'
			'IoU: {}\n'
			'mean IoU: {:.1f}').format(
				acc_global.item() * 100,
				['{:.1f}'.format(i) for i in (acc * 100).tolist()],
				['{:.1f}'.format(i) for i in (iu * 100).tolist()],
				iu.mean().item() * 100)