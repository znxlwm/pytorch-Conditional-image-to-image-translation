import os, time, pickle, argparse, networks, utils, itertools
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dataset', help='dataset')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layers for generator')
parser.add_argument('--img_size', type=int, default=64, help='input image size')
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

# results save path
if not os.path.isdir(os.path.join(args.dataset_name + '_results', 'img')):
    os.makedirs(os.path.join(args.dataset_name + '_results', 'img'))
if not os.path.isdir(os.path.join(args.dataset_name + '_results', 'model')):
    os.makedirs(os.path.join(args.dataset_name + '_results', 'model'))

# data_loader
transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader_A = utils.data_load(os.path.join('data', args.dataset), 'trainA', transform, args.batch_size, shuffle=True, drop_last=True)
train_loader_B = utils.data_load(os.path.join('data', args.dataset), 'trainB', transform, args.batch_size, shuffle=True, drop_last=True)
test_loader_A = utils.data_load(os.path.join('data', args.dataset), 'testA', transform, 1, shuffle=True, drop_last=True)
test_loader_B = utils.data_load(os.path.join('data', args.dataset), 'testB', transform, 1, shuffle=True, drop_last=True)

# network
En_A = networks.encoder(in_nc=args.in_ngc, nf=args.ngf, img_size=args.img_size).to(device)
En_B = networks.encoder(in_nc=args.in_ngc, nf=args.ngf, img_size=args.img_size).to(device)
De_A = networks.decoder(out_nc=args.out_ngc, nf=args.ngf).to(device)
De_B = networks.decoder(out_nc=args.out_ngc, nf=args.ngf).to(device)
Disc_A = networks.discriminator(in_nc=args.in_ndc, out_nc=args.out_ndc, nf=args.ndf, img_size=args.img_size).to(device)
Disc_B = networks.discriminator(in_nc=args.in_ndc, out_nc=args.out_ndc, nf=args.ndf, img_size=args.img_size).to(device)
En_A.train()
En_B.train()
De_A.train()
De_B.train()
Disc_A.train()
Disc_B.train()
print('---------- Networks initialized -------------')
utils.print_network(En_A)
utils.print_network(En_B)
utils.print_network(De_A)
utils.print_network(De_B)
utils.print_network(Disc_A)
utils.print_network(Disc_B)
print('-----------------------------------------------')

# loss
BCE_loss = nn.BCELoss().to(device)
L1_loss = nn.L1Loss().to(device)

# Adam optimizer
Gen_optimizer = optim.Adam(itertools.chain(En_A.parameters(), De_A.parameters(), En_B.parameters(), De_B.parameters()), lr=args.lrG, betas=(args.beta1, args.beta2))
Disc_A_optimizer = optim.Adam(Disc_A.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
Disc_B_optimizer = optim.Adam(Disc_B.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

train_hist = {}
train_hist['Disc_A_loss'] = []
train_hist['Disc_B_loss'] = []
train_hist['Gen_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []
print('training start!')
start_time = time.time()
real = torch.ones(args.batch_size, 1, 1, 1).to(device)
fake = torch.zeros(args.batch_size, 1, 1, 1).to(device)
for epoch in range(args.train_epoch):
    epoch_start_time = time.time()
    En_A.train()
    En_B.train()
    De_A.train()
    De_B.train()
    Disc_A_losses = []
    Disc_B_losses = []
    Gen_losses = []
    iter = 0
    for (A, _), (B, _) in zip(train_loader_A, train_loader_B):
        A, B = A.to(device), B.to(device)

        # train Disc_A & Disc_B
        # Disc real loss
        Disc_A_real = Disc_A(A)
        Disc_A_real_loss = BCE_loss(Disc_A_real, real)

        Disc_B_real = Disc_B(B)
        Disc_B_real_loss = BCE_loss(Disc_B_real, real)

        # Disc fake loss
        in_A, sp_A = En_A(A)
        in_B, sp_B = En_B(B)

        # De_A == B2A decoder, De_B == A2B decoder
        B2A = De_A(in_B + sp_A)
        A2B = De_B(in_A + sp_B)

        Disc_A_fake = Disc_A(B2A)
        Disc_A_fake_loss = BCE_loss(Disc_A_fake, fake)

        Disc_B_fake = Disc_B(A2B)
        Disc_B_fake_loss = BCE_loss(Disc_B_fake, fake)

        Disc_A_loss = Disc_A_real_loss + Disc_A_fake_loss
        Disc_B_loss = Disc_B_real_loss + Disc_B_fake_loss

        Disc_A_optimizer.zero_grad()
        Disc_A_loss.backward(retain_graph=True)
        Disc_A_optimizer.step()

        Disc_B_optimizer.zero_grad()
        Disc_B_loss.backward(retain_graph=True)
        Disc_B_optimizer.step()

        train_hist['Disc_A_loss'].append(Disc_A_loss.item())
        train_hist['Disc_B_loss'].append(Disc_B_loss.item())
        Disc_A_losses.append(Disc_A_loss.item())
        Disc_B_losses.append(Disc_B_loss.item())

        # train Gen
        # Gen adversarial loss
        in_A, sp_A = En_A(A)
        in_B, sp_B = En_B(B)

        B2A = De_A(in_B + sp_A)
        A2B = De_B(in_A + sp_B)

        Dist_A_fake = Disc_A(B2A)
        Gen_A_fake_loss = BCE_loss(Disc_A_fake, real)

        Disc_B_fake = Disc_B(A2B)
        Gen_B_fake_loss = BCE_loss(Disc_B_fake, real)

        # Gen Dual loss
        in_A_hat, sp_B_hat = En_B(A2B)
        in_B_hat, sp_A_hat = En_A(B2A)

        A_hat = De_A(in_A_hat + sp_A)
        B_hat = De_B(in_B_hat + sp_B)

        Gen_gan_loss = Gen_A_fake_loss + Gen_B_fake_loss
        Gen_dual_loss = L1_loss(A_hat, A.detach()) ** 2 + L1_loss(B_hat, B.detach()) ** 2
        Gen_in_loss = L1_loss(in_A_hat, in_A.detach()) ** 2 + L1_loss(in_B_hat, in_B.detach()) ** 2
        Gen_sp_loss = L1_loss(sp_A_hat, sp_A.detach()) ** 2 + L1_loss(sp_B_hat, sp_B.detach()) ** 2

        Gen_loss = Gen_A_fake_loss + Gen_B_fake_loss + Gen_dual_loss + Gen_in_loss + Gen_sp_loss

        Gen_optimizer.zero_grad()
        Gen_loss.backward()
        Gen_optimizer.step()

        train_hist['Gen_loss'].append(Gen_loss.item())
        Gen_losses.append(Gen_loss.item())

        iter += 1

    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)
    print(
        '[%d/%d] - time: %.2f, Disc A loss: %.3f, Disc B loss: %.3f, Gen loss: %.3f' % (
            (epoch + 1), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Disc_A_losses)),
            torch.mean(torch.FloatTensor(Disc_B_losses)), torch.mean(torch.FloatTensor(Gen_losses)),))


    with torch.no_grad():
        En_A.eval()
        En_B.eval()
        De_A.eval()
        De_B.eval()
        n = 0
        for (A, _), (B, _) in zip(test_loader_A, test_loader_B):
            A, B = A.to(device), B.to(device)

            in_A, sp_A = En_A(A)
            in_B, sp_B = En_B(B)

            B2A = De_A(in_B + sp_A)
            A2B = De_B(in_A + sp_B)

            result = torch.cat((A[0], B[0], A2B[0], B2A[0]), 2)
            path = os.path.join(args.dataset_name + '_results', 'img', str(epoch+1) + '_epoch_' + args.dataset_name + '_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            n += 1

        torch.save(En_A.state_dict(), os.path.join(args.dataset_name + '_results', 'model', 'En_A_param_latest.pkl'))
        torch.save(En_B.state_dict(), os.path.join(args.dataset_name + '_results', 'model', 'En_B_param_latest.pkl'))
        torch.save(De_A.state_dict(), os.path.join(args.dataset_name + '_results', 'model', 'De_A_param_latest.pkl'))
        torch.save(De_B.state_dict(), os.path.join(args.dataset_name + '_results', 'model', 'De_B_param_latest.pkl'))
        torch.save(Disc_A.state_dict(), os.path.join(args.dataset_name + '_results', 'model', 'Disc_A_param_latest.pkl'))
        torch.save(Disc_B.state_dict(), os.path.join(args.dataset_name + '_results', 'model', 'Disc_B_param_latest.pkl'))


        if (epoch+1) % 50 == 0:
            torch.save(En_A.state_dict(),
                       os.path.join(args.dataset_name + '_results', 'model', 'En_A_param_' + str(epoch+1) + '.pkl'))
            torch.save(En_B.state_dict(),
                       os.path.join(args.dataset_name + '_results', 'model', 'En_B_param_' + str(epoch+1) + '.pkl'))
            torch.save(De_A.state_dict(),
                       os.path.join(args.dataset_name + '_results', 'model', 'De_A_param_' + str(epoch+1) + '.pkl'))
            torch.save(De_B.state_dict(),
                       os.path.join(args.dataset_name + '_results', 'model', 'De_B_param_' + str(epoch+1) + '.pkl'))
            torch.save(Disc_A.state_dict(),
                       os.path.join(args.dataset_name + '_results', 'model', 'Disc_A_param_' + str(epoch+1) + '.pkl'))
            torch.save(Disc_B.state_dict(),
                       os.path.join(args.dataset_name + '_results', 'model', 'Disc_B_param_' + str(epoch+1) + '.pkl'))

total_time = time.time() - start_time
train_hist['total_time'].append(total_time)

print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
print("Training finish!... save training results")
with open(os.path.join(args.dataset_name + '_results',  'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)
