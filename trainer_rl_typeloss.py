import time
from argparse import ArgumentParser         ##python library for providing flexibilty to change arguments value(especially for command line interface)

##optimizers shape and mold your model into its most accurate possible form by futzing with the weights. The loss function is 
##the guide to the terrain, telling the optimizer when it's moving in the right or wrong direction
import torch.optim as optim                 
import torch.utils.data                     ##Combines a dataset and a sampler, and provides an iterable over the given dataset
from data.load_data import load_data,Dataset    ##using data folder ,also importing Dataset class from data folder
from data import preprocess                      ##using data folder
from model.model_b3_p import Reab3p16            ##using model folder
from model.model_plusMLP import WildRelationNet   
from rl.ddpg import *                             ##using rl folder
from rl.MADDPG import *
from rl.help_function import *                   ##using rl folder
from rl.qlearning import *                       ##using rl folder
import utils                                   ##for image summary
from tensorboard import TensorBoard
from rl.Agent import Agent

##Class labels: Different categories(actions) (logic combn of class) of training sample in the dataset ex: fig1. "position and" 
code = ['shape', 'line', "color", 'number', 'position', 'size',                  
        'type', 'progression', "xor", "or", 'and', 'consistent_union']
logger=utils.get_logger()

##initializing weigths 
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
##It's a function, that can be applied to the whole network and initialize corresponding layer accordingly(in this case - convolution,batchNorm and linear ).
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

        
def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

##function to save the state of model after training on one batch
def save_state(state, path):              
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path+'.pt')

        
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

##Learning Rate is an important hyperpa
eter in Gradient Descent. Its value determines how fast the Neural Network would..
##converge to minima. Function to adjust the larning rate
def adjust_learning_rate(optimizer, epoch, lr_steps,n):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.2

    if n>1:
        for param_group in optimizer.module.param_groups:
            param_group['lr'] = decay * param_group['lr']
            print(("epoch %d : lr=%.5f") % (epoch,  param_group['lr']))
            if epoch>15:
                param_group['momentum'] = 0.9
                param_group['weight_decay'] = decay * param_group['lr']
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']
            param_group['weight_decay'] = decay * param_group['lr']
            print(("epoch %d : lr=%.5f") % (epoch, param_group['lr']))
            if epoch>15:
                param_group['momentum'] = 0.9
def main(args):

    # Step 1: init data folders
    '''if os.path.exists('save_state/'+args.regime+'/normalization_stats.pkl'):           ##to load raw data and preprocess it 
        print('Loading normalization stats')
        x_mean, x_sd = misc.load_file('save_state/'+args.regime+'/normalization_stats.pkl')
    else:
        x_mean, x_sd = preprocess.save_normalization_stats(args.regime)
        print('x_mean: %.3f, x_sd: %.3f' % (x_mean, x_sd))'''

    val_loader=load_data(args, "val")              ##loading already preprocessed validation/testing data 

    tb=TensorBoard(args.model_dir)                ##The model_dir arguments represents the directory to save model parameters, graph and etc. This can also be used to 
                                                  ##load checkpoints from the directory into a estimator to continue training a previously saved model.

    # Step 2: init neural networks
    print("network is:",args.net)
    if args.net == 'Reab3p16':                ##if want to use model Reab3p16
        model = Reab3p16(args)
    elif args.net=='RN_mlp':                  ##if want to use model WildRelationNet
        model =WildRelationNet()
    if args.gpunum > 1:                        
        model = nn.DataParallel(model, device_ids=range(args.gpunum)) ##The nn package defines a set of Modules, which you can think of as a neural network layer that has produces output from 
                                                                       ##input and may have some trainable weights.
                                                                    ##when more than one gpu, want to save model weights using DataParrallel module prefix
    weights_path = args.path_weight+"/"+args.load_weight               ##saved weigths of model 

    if os.path.exists(weights_path) and args.restore:             ##pretrained weights
        pretrained_dict = torch.load(weights_path)                 ##pretrained_dict is the state dictionary of the pre-trained model available
        model_dict = model.state_dict()                           ## https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.htmlA state_dict is an integral entity 
        pretrained_dict1 = {}                                      ##..if you are interested in saving or loading models from PyTorch
        for k, v in pretrained_dict.items():                      ##filter out unnecessary keys k
            if k in model_dict:                                   ##only when keys match(like conv2D..and so forth)
                pretrained_dict1[k] = v
                #print(k)                   
        model_dict.update(pretrained_dict1)                        ##overwrite entries in the existing state dict 
        model.load_state_dict(model_dict)                          ##load the new state dict, new weights

        print('load weight')

    style_raven={65:0, 129:1, 257:2, 66:3, 132:4, 36:5, 258:6, 136:7, 264:8, 72:9, 130:10    ##dictionary(key:value pair of      
         , 260:11, 40:12, 34:13, 49:14, 18:15, 20:16, 24:17}

##After setting weights using optimizer for training.

##The standard way in PyTorch to train a model in multiple GPUs is to use nn.DataParallel which copies the model to the GPUs 
##and during training splits the batch among them and combines the individual outputs.
##model.cuda() by default will send your model to the "current device"

#If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it. Parameters of a model 
#after .cuda() will be different objects with those before the call.

##A very popular technique that is used along with SGD is called Momentum. Instead of using only the gradient of the current 
##step to guide the search, momentum also accumulates the gradient of the past steps to determine the direction to go
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.mo, weight_decay=5e-4) ##Adam has convergence problems that often SGD + momentum can converge better 
                                                                               ##with longer training time. We often see a lot of papers in 2018 and 2019 were still using SGD
    if args.gpunum>1:
        optimizer = nn.DataParallel(optimizer, device_ids=range(args.gpunum))
                                  ##setting iter-count and epoch to 1 before starting training
    iter_count = 1               ## number of batches of data the algorithm has seen (or simply the number of passes the algorithm has done on the dataset)
    epoch_count = 1              ##number of times a learning algorithm sees the complete dataset 
    #iter_epoch=int(len(train_files) / args.batch_size)
    print(time.strftime('%H:%M:%S', time.localtime(time.time())), 'training')
    style_raven_len = len(style_raven)  ##length of  style raven dict
    
    if args.rl_style=="dqn":     ##calling reinforcemt model for training
        dqn = DQN()                ##if want to use dqn model
    elif args.rl_style=="ddpg":    ##if want to use ddpg model (aiming to use this)
        ram = MemoryBuffer(1000)   
        ddpg = Trainer(style_raven_len*4+2, style_raven_len, 1, ram)        ##creating an instance of Trainer class defined  in rl folder (ddpg.py) why style_raven_len*4+2? 
    elif args.rl_style = "maddpg":
        ram = Buffer(1000)
        agents = _init_agents()
        maddpg = Trainer(style_raven_len*4+2, style_raven_len, 1, ram)
    alpha_1=0.1

    if args.rl_style=="dqn":
        a = dqn.choose_action([0.5] * 3)  # TODO
    elif args.rl_style=="ddpg":
        action_ = ddpg.get_exploration_action(np.zeros([style_raven_len*4+2]).astype(np.float32),alpha_1) ##calling exploration which returns action? 
    elif args.rl_style == "maddpg":
        noise   = 0.1
        epsilon = 0.1
    if args.type_loss:loss_fn=nn.BCELoss()                      ##Creates a criterion that measures the Binary Cross Entropy between the target and the output.
    best_acc=0.0                                                ##setting accuracy to 0.0
    while True:                                                ##loop(train)  until
        since=time.time()
        
        if args.rl_style == "maddpg":
            for agent_id, agent in enumerate(agents):
                action_ = agent.select_action(s[agent_id], noise, epsilon)
                
        noise = max(0.05, noise - 0.0000005)
        epsilon = max(0.05, epsilon - 0.0000005)           
        
        print(action_)                                            
        for i in range(style_raven_len):                
            tb.scalar_summary("action/a"+str(i), action_[i], epoch_count) ##saving summary such as poch counts and actions

        data_files = preprocess.provide_data(args.regime, style_raven_len, action_,style_raven) 

        train_files = [data_file for data_file in data_files if 'train' in data_file]               #creating a list of training files
        print("train_num:", len(train_files))
    
        ##torch.utils.data.DataLoader` supports both map-style and iterable-style datasets with single- or multi-process loading,
        ##customizing loading order and optional automatic batching (collation) and memory pinning
        ##shuffle true because we want independent B training batches from Dataset
        train_loader = torch.utils.data.DataLoader(Dataset(args,train_files), batch_size=args.batch_size, shuffle=True,  
                                                   num_workers=args.numwork)
        model.train()                      ##start training model
        iter_epoch = int(len(train_files) / args.batch_size)         ##setting iteration count for total dataset
        acc_part_train=np.zeros([style_raven_len,2]).astype(np.float32)       ##defining variable for saving part accuracy while training

        mean_loss_train= np.zeros([style_raven_len, 2]).astype(np.float32)     ##defining variable for saving mean loss while training
        loss_train=0
        for x, y,style,me in train_loader:                              
            if x.shape[0]<10:                             ##x.shape[0] will give the number of rows in an array  (10 by 1024 2D array)                 
                print(x.shape[0])
                break                                                            
            x, y ,meta = Variable(x).cuda(), Variable(y).cuda(), Variable(me).cuda()  ##Components are accessible as variable.x,  variable.y,  variable.z
            if args.gpunum > 1:                                                        
                optimizer.module.zero_grad()             ##to set the gradient of the parameters in the model to 0, module beacause DataParallel
            else:
                optimizer.zero_grad()                    ## same as above set the gradient of the parameters to zero
            if args.type_loss:
                pred_train, pred_meta= model(x)              ##applying model to x where x is from training data
            else:
                pred_train = model(x)                        ##x is images y is actual label/category
            loss_ = F.nll_loss(pred_train, y,reduce=False)     ##calculating loss occurred while training
            loss=loss_.mean() if not args.type_loss else loss_.mean()+10*loss_fn(pred_meta,meta)##If your loss is not a scalar value, then you should certainly use either 
            loss.backward()             ##loss.mean() or loss.sum() to convert it to a scalar before calling the backward. Otherwise, it will cause an error
        
        #When you call loss.backward(), all it does is compute gradient of loss w.r.t all the parameters in loss that have 
        ##requires_grad = True and store them in parameter.grad attribute for every parameter.
        ##optimizer.step() updates all the parameters based on parameter.grad
            if args.gpunum > 1:
                optimizer.module.step()      ##module for DataParallel
            else:
                optimizer.step()
            iter_count += 1                ##update iter-count by 1 evrytime
            pred = pred_train.data.max(1)[1]  
            correct = pred.eq(y.data).cpu()       ##compare actual and predicted category
            loss_train+=loss.item()               ##The average of the batch losses will give you an estimate of the “epoch loss” during training.
            for num, style_pers in enumerate(style):
                style_pers = style_pers[:-4].split("/")[-1].split("_")[3:]
                for style_per in style_pers:
                    style_per=int(style_per)
                    if correct[num] == 1:
                        acc_part_train[style_per, 0] += 1
                    acc_part_train[style_per, 1] += 1
                    #mean_pred_train[style_per,0] += pred_train[num,y[num].item()].data.cpu()
                    #mean_pred_train[style_per, 1] += 1
                    mean_loss_train[style_per,0] += loss_[num].item()
                    mean_loss_train[style_per, 1] += 1
            accuracy_total = correct.sum() * 100.0 / len(y)       ####calc accuracy 

            if iter_count %10 == 0:                        ##do this for 10 iterations
                iter_c = iter_count % iter_epoch
                print(time.strftime('%H:%M:%S', time.localtime(time.time())),
                      ('train_epoch:%d,iter_count:%d/%d, loss:%.3f, acc:%.1f') % (
                      epoch_count, iter_c, iter_epoch, loss, accuracy_total))
                tb.scalar_summary("train_loss",loss,iter_count)               ##saving train loss to summary
                
        loss_train=loss_train/len(train_files)                             ##The average of the batch losses will give you an estimate of the “epoch loss” during training.
        #mean_pred_train=[x[0]/ x[1] for x in mean_pred_train]
        mean_loss_train=[x[0]/ x[1] for x in mean_loss_train]
        acc_part_train = [x[0] / x[1] if x[1]!=0 else 0  for x in acc_part_train]
        print(acc_part_train)
        if epoch_count %args.lr_step ==0:                 ##adjusting learning rate after  30 epochs
            print("change lr")
            adjust_learning_rate(optimizer, epoch_count, args.lr_step,args.gpunum)
        time_elapsed = time.time() - since
        print('train epoch in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, time_elapsed // 60 % 60, time_elapsed % 60))
        #acc_p=np.array([x[0]/x[1] for x in acc_part])
        #print(acc_p)
        with torch.no_grad():
            model.eval()             ##evaluating model 
            accuracy_all = []
            iter_test=0
            acc_part_val = np.zeros([style_raven_len, 2]).astype(np.float32)
            for x, y, style,me in val_loader:             ##using validation data
                iter_test+=1
                x, y = Variable(x).cuda(), Variable(y).cuda()
                pred,_ = model(x)
                pred = pred.data.max(1)[1]
                correct = pred.eq(y.data).cpu().numpy()
                accuracy = correct.sum() * 100.0 / len(y)   ##accuracy is calc basd on how many labels match
                for num, style_pers in enumerate(style):
                    style_pers = style_pers[:-4].split("/")[-1].split("_")[3:]
                    for style_per in style_pers:
                        style_per = int(style_per)
                        if correct[num] == 1:
                            acc_part_val[style_per, 0] += 1
                        acc_part_val[style_per, 1] += 1
                accuracy_all.append(accuracy)                    ##append to accuracy list

                # if iter_test % 10 == 0:
                #
                #     print(time.strftime('%H:%M:%S', time.localtime(time.time())),
                #           ('test_iter:%d, acc:%.1f') % (
                #               iter_test, accuracy))

        accuracy_all = sum(accuracy_all) / len(accuracy_all)              ##total accuracy is calculated 
        acc_part_val = [x[0] / x[1] if x[1]!=0 else 0 for x in acc_part_val ]
        baseline_rl=70                                          ##baseline for accuracy
        reward=np.mean(acc_part_val)*100-baseline_rl          ##calculating reward using val accuracy
        tb.scalar_summary("valreward", reward,epoch_count)        ##saving summary
        action_list=[x for x in action_]
        cur_state=np.array(acc_part_val+acc_part_train+action_list+mean_loss_train ##saving all calc in currnt state
                           +[loss_train]+[epoch_count]).astype(np.float32)
        #np.expand_dims(, axis=0)
        if args.rl_style == "dqn":
            a = dqn.choose_action(cur_state)  # TODO
        elif args.rl_style == "ddpg":                              ##passing current state to rl model's get_exploration_action
            a = ddpg.get_exploration_action(cur_state,alpha_1)
        elif args.rl_style == "maddpg":
            for agent_id, agent in enumerate(agents):
                a = agent.select_action(cur_state[agent_id], noise, epsilon)

        if alpha_1<1:
            alpha_1+=0.005#0.1
        if epoch_count > 1:                                ##saving  last state and current state ,reward in memory  for epoch >1
            if args.rl_style == "dqn":dqn.store_transition(last_state, a, reward , cur_state)
            elif args.rl_style == "ddpg":ram.add(last_state, a, reward, cur_state)
            elif args.rl_style == "maddpg":ram.store_episode(last_state, a, reward, cur_state)


        if epoch_count > 1:
            if args.rl_style == "dqn":dqn.learn()
            elif args.rl_style == "ddpg":loss_actor, loss_critic=ddpg.optimize()      ##using rl ddpg model's optimize function to for teaching
            elif args.rl_style == "maddpg":
                losses = []
                transitions = Buffer.sample(args.batch_size)
                for agent in agents:
                    other_agents = agents.copy()
                    other_agents.remove(agent)
                    loss_actor, loss_critic = agent.optimize(transitions, other_agents)
                    losses.append([loss_actor, loss_critic])
                
            print('------------------------------------')
            print('learn q learning')
            print('------------------------------------')
            tb.scalar_summary("loss_actor", loss_actor, epoch_count)
            tb.scalar_summary("loss_critic", loss_critic, epoch_count)


        last_state=cur_state
        time_elapsed = time.time() - since
        print('test epoch in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, time_elapsed // 60 % 60, time_elapsed % 60))
        print('------------------------------------')
        print(('epoch:%d, acc:%.1f') % (epoch_count, accuracy_all))
        print('------------------------------------')
        if accuracy_all>best_acc:                                        ##save the best accuracy obtained from val data as best accuracy for next epoch
            best_acc=max(best_acc,accuracy_all)
            #ddpg.save_models(args.model_dir + '/', epoch_count)
            save_state(model.state_dict(), args.model_dir + "/epochbest")             ##saving the current state
        epoch_count += 1                                                         ##increasing  epoch count by 1
        if epoch_count%20==0:              ##Do this for 20 epochs for complete dataset 
            print("save weights")
            madddpg.save_model(epoch_count)                  ##saving the model
            save_state(model.state_dict(), args.model_dir+"/epoch"+str(epoch_count))
        #if epoch_count == 400:
        #        print(f"we have reached to epoch {epoch_count}!")
        #        break



