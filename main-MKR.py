from src.MKR import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=30, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=5, help='embedding size')
    # parser.add_argument('--L', type=int, default=1, help='L')
    # parser.add_argument('--T', type=int, default=1, help='T')
    # parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=30, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=20, help='embedding size')
    # parser.add_argument('--L', type=int, default=1, help='L')
    # parser.add_argument('--T', type=int, default=3, help='T')
    # parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
    #
    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=30, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=10, help='embedding size')
    # parser.add_argument('--L', type=int, default=1, help='L')
    # parser.add_argument('--T', type=int, default=1, help='T')
    # parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=10, help='embedding size')
    parser.add_argument('--L', type=int, default=1, help='L')
    parser.add_argument('--T', type=int, default=1, help='T')
    parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()

    train(args, True)

'''
music	train_auc: 0.891 	 train_acc: 0.776 	 eval_auc: 0.823 	 eval_acc: 0.750 	 test_auc: 0.823 	 test_acc: 0.757 		[0.16, 0.28, 0.47, 0.51, 0.51, 0.55, 0.57, 0.58]
book	train_auc: 0.878 	 train_acc: 0.758 	 eval_auc: 0.751 	 eval_acc: 0.691 	 test_auc: 0.749 	 test_acc: 0.689 		[0.09, 0.15, 0.27, 0.31, 0.31, 0.38, 0.39, 0.42]
ml	train_auc: 0.934 	 train_acc: 0.835 	 eval_auc: 0.897 	 eval_acc: 0.798 	 test_auc: 0.897 	 test_acc: 0.801 		[0.17, 0.27, 0.51, 0.53, 0.53, 0.62, 0.67, 0.7]
yelp	train_auc: 0.878 	 train_acc: 0.772 	 eval_auc: 0.832 	 eval_acc: 0.752 	 test_auc: 0.832 	 test_acc: 0.754 		[0.11, 0.17, 0.34, 0.36, 0.36, 0.42, 0.44, 0.49]
'''