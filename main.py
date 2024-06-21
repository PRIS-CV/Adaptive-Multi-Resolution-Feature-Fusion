import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.optim as optim
import argparse
import logging
import learn2learn as l2l

from tools import *
from network import *

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Training code for Birds_ResNet_448')

parser.add_argument('--data_path', default='/data/yangyuqi/CUB_200_2011', type=str,
                    help='path to the dataset (CUB-200-2011)')
parser.add_argument('--work_dirs', default='/home/yangyuqi/cross/result', type=str,
                    help='path to save log and checkpoints')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='the number of training epoch')
parser.add_argument('--batch_size', default=64, type=int, help='the batch size')
parser.add_argument('--maml', '-T', action='store_false', default=True, help='meta learning')
parser.add_argument('--num_classes', default=200, type=int, help='the number of classes')


args = parser.parse_args()

#--------------------------------------------------------------------------------
logging.info(args)

exp_dir = args.work_dirs

try:
    os.stat(exp_dir)
except:
    os.makedirs(exp_dir)
logging.info("OPENING " + exp_dir + '/results_train.csv')
logging.info("OPENING " + exp_dir + '/results_test.csv')


results_train_file = open(exp_dir + '/results_train.csv', 'w')
results_train_file.write('epoch, train_acc_low, train_acc_high, train_acc_fusion, train_loss\n')
results_train_file.flush()

results_test_file = open(exp_dir + '/results_test.csv', 'w')
results_test_file.write('epoch, test_acc_low, test_acc_high, test_acc_fusion, test_loss\n')
results_test_file.flush()

#----------------------------------------------------------------------------------
use_cuda = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss()

print('==> Preparing data..')
trainloader, testloader = dataset(args.data_path)

print('==> Building model..')
net = models.resnet50(pretrained=True)
net = Model_Ours(net, 512, args.num_classes)

#-----------------------------------------------------------------------------------
print('==> Start Training')

if use_cuda:
    net.classifier.cuda()
    net.classifier_low.cuda()
    net.classifier_high.cuda()

    net.features.cuda()
    net.Gate.cuda()

    net.avg_pool.cuda()

    net.expert_low.cuda()
    net.expert_high.cuda()

    net.classifier = torch.nn.DataParallel(net.classifier)
    net.classifier_low = torch.nn.DataParallel(net.classifier_low)
    net.classifier_high = torch.nn.DataParallel(net.classifier_high)

    net.features = torch.nn.DataParallel(net.features)
    net.Gate = torch.nn.DataParallel(net.Gate)

    net.avg_pool = torch.nn.DataParallel(net.avg_pool)

    net.expert_low = torch.nn.DataParallel(net.expert_low)
    net.expert_high = torch.nn.DataParallel(net.expert_high)
    cudnn.benchmark = True

max_test_acc = 0
lr = args.lr

for epoch in range(1, args.epoch+1):

    if args.maml != False:

        optimizer = optim.SGD([
                        {'params': net.classifier_low.parameters(), 'lr': 0.01},
                        {'params': net.classifier_high.parameters(), 'lr': 0.01},
                        {'params': net.classifier.parameters(), 'lr': 0.01},
                        {'params': net.expert_low.parameters(), 'lr': 0.01},
                        {'params': net.expert_high.parameters(), 'lr': 0.01},
                        {'params': net.features.parameters(),   'lr': 0.001}
                        
                     ], 
                      momentum=0.9, weight_decay=5e-4)

        optimizer_maml = optim.SGD([
                        {'params': net.Gate.parameters(), 'lr': 0.01},
                        
                     ], 
                      momentum=0.9, weight_decay=5e-4)

        optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer.param_groups[3]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer.param_groups[4]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer.param_groups[5]['lr'] =  cosine_anneal_schedule(epoch, lr)/10

        optimizer_maml.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, lr)

        maml_model = l2l.algorithms.MAML(net, lr=cosine_anneal_schedule(epoch, lr), first_order=True, allow_unused=True, allow_nograd=False)
        train_acc_low, train_acc_high, train_acc_fusion, train_loss = train_maml(epoch, net, trainloader, optimizer, optimizer_maml, maml_model)
    else:

        optimizer_ = optim.SGD([
                        {'params': net.classifier_low.parameters(), 'lr': 0.01},
                        {'params': net.classifier_high.parameters(), 'lr': 0.01},
                        {'params': net.classifier.parameters(), 'lr': 0.01},
                        {'params': net.expert_low.parameters(), 'lr': 0.01},
                        {'params': net.expert_high.parameters(), 'lr': 0.01},
                        {'params': net.Gate.parameters(), 'lr': 0.01},
                        {'params': net.features.parameters(),   'lr': 0.001}
                        
                     ], 
                      momentum=0.9, weight_decay=5e-4)

        optimizer_.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer_.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer_.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer_.param_groups[3]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer_.param_groups[4]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer_.param_groups[5]['lr'] =  cosine_anneal_schedule(epoch, lr)
        optimizer_.param_groups[6]['lr'] =  cosine_anneal_schedule(epoch, lr)/10

        train_acc_low, train_acc_high, train_acc_fusion, train_loss = train(epoch, net, trainloader, optimizer_)

    logging.info('Iteration %d, train_acc_low = %.5f, train_acc_high = %.5f, train_acc_fusion = %.5f,train_loss = %.6f' % (epoch, train_acc_low, train_acc_high, train_acc_fusion, train_loss))
    results_train_file.write('%d, %.4f, %.4f, %.4f, %.4f\n' % (epoch, train_acc_low, train_acc_high, train_acc_fusion, train_loss))
    results_train_file.flush()

    test_acc_low, test_acc_high, test_acc_fusion, test_loss = test(epoch, net, testloader)
    logging.info('test1, test_acc_low = %.5f, test_acc_high = %.5f, test_acc_fusion = %.5f, test_loss = %.4f' % (test_acc_low, test_acc_high, test_acc_fusion, test_loss))
    results_test_file.write('%d, %.4f, %.4f, %.4f, %.4f\n' % (epoch, test_acc_low, test_acc_high, test_acc_fusion, test_loss))
    results_test_file.flush()

    if test_acc_fusion > max_test_acc:
        max_test_acc = test_acc_fusion
        torch.save(net.state_dict(), args.work_dirs +'/model_best.pth')
    print("max_test_acc", max_test_acc)