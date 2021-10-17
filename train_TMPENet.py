import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
# from learning3d.models import tpccNet

sys.path.append('E:\Mtmpe-Net\TMPENet\tmpenet\models')
import tmpenet
from tmpenet import itmpeNet
import pointnet
from pointnet import PointNet
sys.path.append('E:\Mtmpe-Net\TPCCNet\learning3d\models')
import tpccnet
from tpccnet import tpccNet
sys.path.append('E:\Mtmpe-Net\TPCCNet\learning3d\data_utils')
import dataloaders
from dataloaders import RegistrationData, ModelNet40Data	#dataloaders.py File line 169


def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')



class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()

def evaluate_metrics(TP, FP, FN, TN, gt_tpcc):
	# TP, FP, FN, TN: 		True +ve, False +ve, False -ve, True -ve
	# gt_tpcc:				Ground Truth tpcc [Nt, 1]
	
	accuracy = (TP + TN)/gt_tpcc.shape[1]
	misclassification_rate = (FN + FP)/gt_tpcc.shape[1]
	# Precision: (What portion of positive identifications are actually correct?)  #Determine the accuracy of true 1 and true 0, accounting for the template point cloud
	precision = TP / (TP + FP)
	# Recall: (What portion of actual positives are identified correctly?) #Judging the accuracy of true 1, accounting for the source point cloud
	recall = TP / (TP + FN)

	fscore = (2*precision*recall) / (precision + recall)
	return accuracy, precision, recall, fscore

# Function used to evaluate the predicted tpcc with ground truth tpcc.
def evaluate_tpcc(gt_tpcc, predicted_tpcc, predicted_tpcc_idx):
	# gt_tpcc:					Ground Truth tpcc [Nt, 1]
	# predicted_tpcc:			tpcc predicted by network [Nt, 1]
	# predicted_tpcc_idx:		Point indices chosen by network [Ns, 1]

	if torch.is_tensor(gt_tpcc): gt_tpcc = gt_tpcc.detach().cpu().numpy()
	if torch.is_tensor(gt_tpcc): predicted_tpcc = predicted_tpcc.detach().cpu().numpy()
	if torch.is_tensor(predicted_tpcc_idx): predicted_tpcc_idx = predicted_tpcc_idx.detach().cpu().numpy()
	gt_tpcc, predicted_tpcc, predicted_tpcc_idx = gt_tpcc.reshape(1,-1), predicted_tpcc.reshape(1,-1), predicted_tpcc_idx.reshape(1,-1)
	
	gt_idx = np.where(gt_tpcc == 1)[1].reshape(1,-1) 				# Find indices of points which are actually in source.    np.where(gt_tpcc == 1)Output the coordinates of the elements that meet the conditions 

	# TP + FP = number of source points.
	TP = np.intersect1d(predicted_tpcc_idx[0], gt_idx[0]).shape[0]			# is inliner and predicted as inlier (True Positive) 		(Find common indices in predicted_tpcc_idx, gt_idx)，Find the same value in two arrays np.intersect1d（）
	FP = len([x for x in predicted_tpcc_idx[0] if x not in gt_idx])			# isn't inlier but predicted as inlier (False Positive)  0→1
	FN = FP															# is inlier but predicted as outlier (False Negative) (due to binary classification) 1→0  The number of true 1s is equal to TP + FP = number of source points The number of is fixed, so how many 0→1 corresponds to the same number 1→0 to maintain the fixed number of 1
	TN = gt_tpcc.shape[1] - gt_idx.shape[1] - FN 					# is outlier and predicted as outlier (True Negative)  即TN=（ gt_tpcc.shape[1] - gt_idx.shape[1] ）- FN    gt_tpcc.shape[1] - gt_idx.shape[1] The number of true zeros is also a fixed value, minus the false zero of the wrong judgment is equal to the zero of the right judgment
	return evaluate_metrics(TP, FP, FN, TN, gt_tpcc)

def test_one_epoch(args, model, model2, test_loader):
	model.eval()
	model2.eval()
	test_loss1 = 0.0
	test_loss2=0.0
	pred  = 0.0
	count = 0
	AP_List = []
	GT_Size_List = []
	precision_list = []
	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt, gt_tpcc = data

		template = template.to(args.device)
		source = source.to(args.device)
		igt = igt.to(args.device)				# [source] = [igt]*[template]
		gt_tpcc = gt_tpcc.to(args.device)

		tpcced_template, predicted_tpcc = model(template, source)

		accuracy, precision, recall, fscore = evaluate_tpcc(gt_tpcc, predicted_tpcc, predicted_tpcc_idx = model.tpcc_idx)
		#print("one Precision: ", precision)
		precision_list.append(precision)
		output = model2(tpcced_template, source)

		est_T=output['est_2']
		dim = igt.shape #Get the dimension of the original matrix
		igt=torch.reshape(igt,(dim[0],dim[2]))	#The dimension becomes consistent with est_T
		r=est_T-igt
		z = torch.zeros_like(r) #Generate a matrix of all zeros that is consistent with the dimensionality of the variables in the brackets
		loss_val1= torch.nn.functional.mse_loss(r, z)*1e4

		loss_val=loss_val1

		test_loss2 += loss_val.item()
		count += 1

	test_loss2 = float(test_loss2)/count
	return test_loss2,precision_list

def test(args, model, model2, test_loader, textio):
	test_loss2 ,precision_list= test_one_epoch(args.device, model, model2, test_loader)#,precision_list

def train_one_epoch(args, model, model2, train_loader, optimizer2):
	model.eval()
	model2.train()

	train_loss2= 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(train_loader)):
		template, source, igt, gt_tpcc = data

		template = template.to(args.device)
		source = source.to(args.device)
		igt = igt.to(args.device)					# [source] = [igt]*[template]

		gt_tpcc = gt_tpcc.to(args.device)

		tpcced_template, predicted_tpcc = model(template, source)
				
		if args.loss_fn == 'mse':
			loss_tpcc = torch.nn.functional.mse_loss(predicted_tpcc, gt_tpcc)
		elif args.loss_fn == 'bce':
			loss_tpcc = torch.nn.BCELoss()(predicted_tpcc, gt_tpcc)
		
		output = model2(tpcced_template, source)

		est_T=output['est_2']
		dim = igt.shape #Get the dimension of the original matrix
		igt=torch.reshape(igt,(dim[0],dim[2]))	#The dimension becomes consistent with est_T

		r=est_T-igt
		z = torch.zeros_like(r) #Generate a matrix of all zeros that is consistent with the dimensionality of the variables in the brackets
		loss_val1= torch.nn.functional.mse_loss(r, z)*1e4

		loss_val=loss_val1

		
		optimizer2.zero_grad()
		loss_val.backward()
		optimizer2.step()
		train_loss2 += loss_val.item()
		count += 1

	
	train_loss2 = float(train_loss2)/count
	return train_loss2

def train(args, model, model2, train_loader, test_loader, boardio, textio, checkpoint):

	learnable_params2 = filter(lambda p2: p2.requires_grad, model2.parameters())
	if args.optimizer == 'Adam':

		optimizer2 = torch.optim.Adam(learnable_params2, lr=0.001)
	else:

		optimizer2 = torch.optim.SGD(learnable_params2, lr=0.1)

	if checkpoint is not None:
		
		min_loss2 = checkpoint['min_loss2']			#Combined with the checkpoint function on line 222, it is used to call the saved model to continue training

		optimizer2.load_state_dict(checkpoint['optimizer2'])


	#best_test_loss1 = np.inf
	best_test_loss2 = np.inf

	for epoch in range(args.start_epoch, args.epochs):
		train_loss2 = train_one_epoch(args, model, model2, train_loader, optimizer2)
		test_loss2,precision_list = test_one_epoch(args, model, model2, test_loader)
		print("Mean Precision: ", np.mean(precision_list))
		
		if test_loss2<best_test_loss2:
			best_test_loss2 = test_loss2
			snap2 = {'epoch2': epoch + 1,
		
				'model2': model2.state_dict(),
				'min_loss2': best_test_loss2,
				'optimizer2' : optimizer2.state_dict(),}
			torch.save(snap2, 'checkpoints/%s/models/model_snap2.t7' % (args.exp_name))
			torch.save(model2.state_dict(), 'checkpoints/%s/models/best_model2.t7' % (args.exp_name))

		snap2 = {'epoch2': epoch + 1,
				'model2': model2.state_dict(),
				
				'min_loss2': best_test_loss2,
				'optimizer2' : optimizer2.state_dict(),}
		torch.save(snap2, 'checkpoints/%s/models/model_snap2.t7' % (args.exp_name))
		torch.save(model2.state_dict(), 'checkpoints/%s/models/model2.t7' % (args.exp_name))



		boardio.add_scalar('Train_Loss2', train_loss2, epoch+1)	

		boardio.add_scalar('Test_Loss2', test_loss2, epoch+1)

		boardio.add_scalar('Best_Test_Loss2', best_test_loss2, epoch+1)

		textio.cprint('EPOCH:: %d,  Traininig Loss2: %f,Testing Loss2: %f,  Best Loss2: %f'
					%(epoch+1, train_loss2,  test_loss2,best_test_loss2))

def options():
	parser = argparse.ArgumentParser(description='tpccNet: A Fully-Convolutional Network For Inlier Estimation (Training)')
	parser.add_argument('--exp_name', type=str, default='exp_tpccnet', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')
	
	# settings for input data
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')#The number of target point clouds, such as 2048
	parser.add_argument('--partial_source', default=True, type=bool,
						help='create partial source point cloud in dataset.')
	parser.add_argument('--noise', default=False, type=bool,
						help='Add noise in source point clouds.')		#The default is False, set in dataloader, you can also set the number of source points, such as 468
	parser.add_argument('--outliers', default=False, type=bool,
						help='Add outliers to template point cloud.')		#Default False, set in dataloader

	# settings for PointNet----tmpe
	parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
						help='train pointnet (default: tune)')
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')
	
	# settings for on training

	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=16, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--test_batch_size', default=16, type=int,
						metavar='N', help='test-mini-batch size (default: 8)')
	parser.add_argument('--unseen', default=True, type=bool,
						help='Use first 20 categories for training and last 20 for testing')		#defaut False
	parser.add_argument('--epochs', default=500, type=int,		#Paper setting 300
						metavar='N', help='number of total epochs to run')
	parser.add_argument('--start_epoch', default=0, type=int,
						metavar='N', help='manual epoch number (useful on restarts)')
	parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
						metavar='METHOD', help='name of an optimizer (default: Adam)')
	parser.add_argument('--resume', default='', type=str,
						metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
	parser.add_argument('--pretrained1', default='path to pretrained model file', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')
	parser.add_argument('--loss_fn', default='mse', type=str, choices=['mse', 'bce'])

	args = parser.parse_args()
	return args

def main():
	args = options()

	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
	_init_(args)

	textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
	textio.cprint(str(args))

	trainset = RegistrationData(ModelNet40Data(train=True, num_points=args.num_points, unseen=args.unseen),
								partial_source=args.partial_source, noise=args.noise, outliers=args.outliers)
	testset = RegistrationData(ModelNet40Data(train=False, num_points=args.num_points, unseen=args.unseen),
								partial_source=args.partial_source, noise=args.noise, outliers=args.outliers)
	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
	test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	model = tpccNet()
	model = model.to(args.device)

	ptnet = PointNet(emb_dims=args.emb_dims)
	model2 = itmpeNet(feature_model=ptnet)
	model2 = model2.to(args.device)

	checkpoint = None
	if args.resume:
		assert os.path.isfile(args.resume)
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']		#Here checkpoint is used to call the saved model to continue training

		model2.load_state_dict(checkpoint['model2'])

	if args.pretrained1:
		assert os.path.isfile(args.pretrained1)
		model.load_state_dict(torch.load(args.pretrained1, map_location='cpu'))
	model.to(args.device)

	if args.eval:
		test(args, model, model2, test_loader, textio)
	else:
		train(args, model, model2, train_loader, test_loader, boardio, textio, checkpoint)

if __name__ == '__main__':
	main()