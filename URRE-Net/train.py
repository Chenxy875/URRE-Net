import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import data
import model
import Loss_fuction
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'

	UVRE_net = model.UVRE().cuda()

	UVRE_net.apply(weights_init)
	if config.load_pretrain == True:
	    UVRE_net.load_state_dict(torch.load(config.model_dir))
	train_dataset = data.lowlight_loader(config.IMAGEpath)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)



	L_dcl = Loss_fuction.L_dcl()
	L_fl = Loss_fuction.L_fl()

	L_ecl = Loss_fuction.L_ecl(16,0.6)
	L_isl = Loss_fuction.L_isl()
	L_noi = Loss_fuction.noise_loss()

	optimizer = torch.optim.Adam(UVRE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	UVRE_net.train()

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			enhanced_image_1,enhanced_image,A,n  = UVRE_net(img_lowlight)

			Loss_TV = 200*L_isl(A)
			loss_noise = 50*torch.mean(L_noi(n))
			
			loss_spa = torch.mean(L_fl(enhanced_image, img_lowlight))

			loss_col = 5*torch.mean(L_dcl(enhanced_image))

			loss_exp = 10*torch.mean(L_ecl(enhanced_image))
			
			
			# best_loss
			loss =  Loss_TV + loss_spa + loss_col + loss_exp + loss_noise
			#

			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(UVRE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(UVRE_net.state_dict(), config.model_folder + "Epoch" + str(epoch) + '.pth') 		




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--IMAGEpath', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--VAL_BS', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--model_folder', type=str, default="model/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--model_dir', type=str, default= "model/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.model_folder):
		os.mkdir(config.model_folder)


	train(config)








	
