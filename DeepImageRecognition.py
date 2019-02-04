import time
import sys
import os
import torch
from torch.autograd import Variable
import torchvision
import numpy as np

IMAGE_SIZE = 256
CHANNELS = 1
DIMENSION = 80

LR_THRESHOLD = 1e-7
TRYING_LR = 3
DEGRADATION_TOLERANCY = 7
ACCURACY_TRESHOLD = float(0.0625)


class DeepImageRecognition(object):
    def __init__(self, recognitron,  criterion, optimizer, directory):
        self.recognitron = recognitron
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_gpu = torch.cuda.is_available()
        self.dispersion = 1.0
        config = str(recognitron.__class__.__name__) + '_' + str(recognitron.activation.__class__.__name__) #+ '_' + str(recognitron.norm1.__class__.__name__)
        config += '_' + str(criterion.__class__.__name__)
        config += "_" + str(optimizer.__class__.__name__) #+ "_lr_" + str( optimizer.param_groups[0]['lr'])

        reportPath = os.path.join(directory, config + "/report/")
        flag = os.path.exists(reportPath)
        if flag != True:
            os.makedirs(reportPath)
            print('os.makedirs("reportPath")')

        self.modelPath = os.path.join(directory, config + "/model/")
        flag = os.path.exists(self.modelPath)
        if flag != True:
            os.makedirs(self.modelPath)
            print('os.makedirs("/modelPath/")')

        self.images = os.path.join(directory, config + "/images/")
        flag = os.path.exists(self.images)
        if flag != True:
            os.makedirs(self.images+'/bad/')
            os.makedirs(self.images + '/good/')
            print('os.makedirs("/images/")')

        self.report = open(reportPath  + '/' + config + "_Report.txt", "w")
        _stdout = sys.stdout
        sys.stdout = self.report
        print(config)
        print(recognitron)
        print(criterion)
        self.report.flush()
        sys.stdout = _stdout
        if self.use_gpu :
            self.recognitron = self.recognitron.cuda()

    def __del__(self):
        self.report.close()

    def approximate(self, dataloaders, num_epochs = 20, resume_train = False, dropout_factor=0):
        path = self.modelPath +"/"+ str(self.recognitron.__class__.__name__) +  str(self.recognitron.activation.__class__.__name__)
        if resume_train and os.path.isfile(path + '_Best.pth'):
            print( "RESUME training load Bestrecognitron")
            self.recognitron.load_state_dict(torch.load(path + '_Best.pth'))

        since = time.time()
        best_loss = 10000.0
        best_acc = 0.0
        counter = 0
        i = int(0)
        degradation = 0
        self.recognitron.set_dropout(dropout_factor)
        for epoch in range(num_epochs):
            _stdout = sys.stdout
            sys.stdout = self.report
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            self.report.flush()
            sys.stdout = _stdout
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.recognitron.train(True)
                else:
                    self.recognitron.train(False)

                running_loss = 0.0
                running_corrects = 0
                for data in dataloaders[phase]:
                    inputs, targets = data
                    if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                        targets = Variable(targets.cuda())
                    else:
                        inputs, targets = Variable(inputs), Variable(targets)
                    self.optimizer.zero_grad()

                    outputs = self.recognitron(inputs)
                    diff = torch.abs(targets.data - torch.round(outputs.data))
                    loss = self.criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (1.0 - torch.sum(diff) / float(diff.shape[1] * diff.shape[0])) * inputs.size(0)

                epoch_loss = float(running_loss) / float(len(dataloaders[phase].dataset))
                epoch_acc = float(running_corrects) / float(len(dataloaders[phase].dataset))

                _stdout = sys.stdout
                sys.stdout = self.report
                print('{} Loss: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_loss, epoch_acc))
                self.report.flush()

                sys.stdout = _stdout
                print('{} Loss: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_loss, epoch_acc))
                self.report.flush()

                if phase == 'val' and epoch_acc > best_acc:
                    counter = 0
                    degradation = 0
                    best_acc = epoch_acc
                    print('curent best_acc ', best_acc)
                    self.save('Best')
                else:
                    counter += 1
                    self.save('Regular')

            if counter > TRYING_LR * 2:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    if lr >= LR_THRESHOLD:
                        param_group['lr'] = lr * 0.2
                        print('! Decrease LearningRate !', lr)

                probas = self.recognitron.get_dropout()
                if dropout_factor > 0 and dropout_factor < 1 and probas < 0.99:
                    print('! Increase DropOut value !', probas)
                    probas += 0.1
                    self.recognitron.set_dropout(probas)

                counter = 0
                degradation += 1
            if degradation > DEGRADATION_TOLERANCY:
                print('This is the end! Best val best_acc: {:4f}'.format(best_acc))
                return best_acc

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val best_acc: {:4f}'.format(best_acc))
        return best_acc


    def evaluate(self, test_loader, isSaveImages = True, modelPath=None):
        counter = 0
        if modelPath is not None:
            self.recognitron.load_state_dict(torch.load(modelPath))
            print('load Recognitron model')
        else:
            self.recognitron.load_state_dict(torch.load(self.modelPath +"/"+ str(self.recognitron.__class__.__name__) +  str(self.recognitron.activation.__class__.__name__) + '_Best.pth'))
            print('load BestRecognitron ')
        print(len(test_loader.dataset))
        i = 0
        since = time.time()
        self.recognitron.train(False)
        self.recognitron.eval()
        if self.use_gpu:
            self.recognitron = self.recognitron.cuda()
        running_loss = 0.0
        running_corrects = 0
        for data in test_loader:
            inputs, targets = data

            if self.use_gpu:
                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())
            else:
                inputs, targets = Variable(inputs), Variable(targets)

            outputs = self.recognitron(inputs)
            diff = torch.abs(targets.data - torch.round(outputs.data))
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (1.0 - torch.sum(diff) / float(diff.shape[1] * diff.shape[0])) * inputs.size(0)


            if isSaveImages and test_loader.batch_size == 1:
                result = torch.round(outputs).data.cpu().numpy()
                result = np.squeeze(result)
                indexes = np.nonzero(result)
                image = inputs.clone()
                image = image.data.cpu().float()
                counter = counter + 1
                if float(diff.item()) > 0.5:
                    filename = self.images + '/bad/'
                else:
                    filename = self.images + '/good/'

                filename+= str(counter) + "__" + str(float(diff.item()))+'__.png'
                #torchvision.utils.save_image(image, filename)

        epoch_loss = float(running_loss) / float(len(test_loader.dataset))
        epoch_acc = float(running_corrects) / float(len(test_loader.dataset))

        time_elapsed = time.time() - since

        print('Evaluating complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Loss: {:.4f} Accuracy {:.4f} '.format( epoch_loss, epoch_acc))
        #self.report.flush()

    def save(self, model):
        self.recognitron = self.recognitron.cpu()
        self.recognitron.eval()
        x = Variable(torch.zeros(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        path = self.modelPath +"/"+ str(self.recognitron.__class__.__name__ ) +  str(self.recognitron.activation.__class__.__name__)
        torch_out = torch.onnx._export(self.recognitron, x, path + "_" + model + ".onnx", export_params=True)
        torch.save(self.recognitron.state_dict(), path + "_" + model  + ".pth")

        if self.use_gpu:
            self.recognitron = self.recognitron.cuda()


class MultiLabelLoss(torch.nn.Module):
    def __init__(self):
        super(MultiLabelLoss, self).__init__()
        self.criterion = torch.nn.BCELoss()
        self.penalty = torch.nn.MSELoss()
        self.loss = None

    def forward(self, input, target):
        p = input.data.cpu().numpy()
        t = target.data.cpu().numpy()
        positive_input = Variable(torch.from_numpy(p[np.nonzero(t)]))
        positive_target = Variable(torch.from_numpy(t[np.nonzero(t)]))
        negative_input = Variable(torch.from_numpy(p[np.where(t == 0)]))
        negative_target = Variable(torch.from_numpy(t[np.where(t == 0)]))
        if torch.cuda.is_available():
            positive_input  = positive_input .cuda()
            positive_target = positive_target.cuda()
            negative_input  = negative_input .cuda()
            negative_target = negative_target.cuda()

        bce_loss = self.criterion(input, target)
        false_positive_penalty = self.penalty(positive_input, positive_target)
        false_negative_penalty= self.penalty(negative_input, negative_target)
        self.loss =  bce_loss + 0.1*false_negative_penalty + 0.1*false_positive_penalty
        return self.loss

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)
