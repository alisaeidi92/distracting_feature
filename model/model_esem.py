import torch
import torch.nn.functional as F

from torch import nn

# T-LEN model (2 stream logic embedding network)

class esem(nn.Module):
    def __init__(self):
        super(esem, self).__init__()
        self.NUM_PANELS = 16

        # used to process shape panels
        self.cnn_shape = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # used to process line panels
        self.cnn_line = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # used to get panel embedding of all 8 context panels
        self.cnn_global = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # for shape panels
        self.pre_g_fc_shape = nn.Linear(32 * 9 ** 2, 256)
        self.pre_g_batch_norm_shape= nn.BatchNorm1d(256)

        # for line panels
        self.pre_g_fc_line = nn.Linear(32 * 9 ** 2, 256)
        self.pre_g_batch_norm_line = nn.BatchNorm1d(256)

        # used to get panel embedding of all 8 context panels
        self.pre_g_fc2 = nn.Linear(32 * 9 ** 2, 256)
        self.pre_g_batch_norm2 = nn.BatchNorm1d(256)

        # used to combine shape and line panel embeddings into one
        self.pre_g_fc_fusion = nn.Linear(256*2, 256)
        self.pre_g_batch_fusion = nn.BatchNorm1d(256)

        # used in function g1 to get g-values of row and column triplets
        self.g = nn.Sequential(
            nn.Linear(512+512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256*3),
            nn.BatchNorm1d(256*3),
            nn.ReLU(),
            nn.Linear(256*3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        # used in function g2 to get g-values of non-row and non-column triplets
        self.g2 = nn.Sequential(
            nn.Linear(512 + 512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 3),
            nn.BatchNorm1d(256 * 3),
            nn.ReLU(),
            nn.Linear(256 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        # used in function f to get f-scores
        self.f = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 1)
        )

    # given a shape and line panel, returns a combined embedding
    def comp_panel_embedding(self, panel, panel2):
        # panel: shape
        # panel2: line

        batch_size = panel.shape[0]
        panel = torch.unsqueeze(panel, 1)  # (batch_size, 160, 160) -> (batch_size, 1, 160, 160)
        panel2= torch.unsqueeze(panel2, 1)

        # line panel
        panel_embedding = self.cnn_line(panel2)  # (batch_size, 1, 160, 160) -> (batch_size, 32, 9, 9)
        panel_embedding = panel_embedding.view(batch_size, 32 * 9 ** 2)
        panel_embedding = self.pre_g_fc_line(panel_embedding)
        panel_embedding_line = self.pre_g_batch_norm_line(panel_embedding)

        # shape panel
        panel_embedding = self.cnn_shape(panel)  # (batch_size, 1, 160, 160) -> (batch_size, 32, 9, 9)
        panel_embedding = panel_embedding.view(batch_size, 32 * 9 ** 2)
        panel_embedding = self.pre_g_fc_shape(panel_embedding)
        panel_embedding_shape = self.pre_g_batch_norm_shape(panel_embedding)

        # combine shape and line panel into one
        fusion=torch.cat([panel_embedding_line,panel_embedding_shape],1)
        panel_embedding = self.pre_g_fc_fusion(fusion)
        panel_embedding = self.pre_g_batch_fusion(panel_embedding)
        panel_embedding = F.relu(panel_embedding)

        return panel_embedding

    # outputs row, col, and other triplets that dont need an answer panel
    def panel_comp_obj_pairs(self, objs, batch_size):
        obj_pairses_r =torch.zeros(batch_size, 2, 256*3).cuda()
        obj_pairses_c = torch.zeros(batch_size, 2, 256 * 3).cuda()
        obj_pairses= torch.zeros(batch_size, 54, 256 * 3).cuda()

        count=0
        index=0
        for i in range(8):
            for j in range(i):
                for k in range(j):
                    # if the panels are the first or second row, combine them into one triplet
                    if ((7-i)==0 and ((7-j)==1) and ((7-k)==2)) or((7-i)==3 and ((7-j)==4) and ((7-k)==5)):
                        obj_pairses_r[:, (7-i)//3, :] = torch.cat(
                            [torch.cat([objs[:, 7 - i, :], objs[:, 7 - j, :]], 1), objs[:, 7 - k, :]], 1)
                        count -= 1

                    # if the panels are the first of second column, combine them into one triplet
                    elif ((7-i)==0 and ((7-j)==3) and ((7-k)==6)) or((7-i)==1 and ((7-j)==4) and ((7-k)==7)):
                        obj_pairses_c[:, 7-i, :] = torch.cat(
                            [torch.cat([objs[:, 7 - i, :], objs[:, 7 - j, :]], 1), objs[:, 7 - k, :]], 1)
                        count -= 1
                    
                    # else combine the panels into pairs that need an answer panel
                    else:
                        obj_pairses[:,count,:] = torch.cat([torch.cat([objs[:, 7 - i, :], objs[:, 7 - j, :]],1), objs[:, 7-k, :]], 1)
                    #obj_pairs = torch.cat([torch.unsqueeze( objs[:,7-i,:],1),torch.unsqueeze( objs[:,7-j,:],1)],2)
                    #obj_pairses[:,count,:] = torch.cat([obj_pairs, torch.unsqueeze(objs[:, 7-k, :], 1)], 2)
                    count+=1
        return obj_pairses,obj_pairses_c,obj_pairses_r

    # ouputs row, col, and other triplets that have an answer panel
    # ans: answer panel embedidng
    # pan: embeddings of context panels
    def ans_comp_obj_pairs(self, ans, pan, batch_size):
        obj_pairses_r = torch.zeros(batch_size, 1, 256 * 3).cuda()
        obj_pairses_c = torch.zeros(batch_size, 1, 256 * 3).cuda()

        obj_pairses=torch.zeros(batch_size, 26, 256*3).cuda()
        count=0
        for i in range(8):
            for j in range(i):
                # if its the last column, combine the context panels with the ans panel to make a triplet
                if (7-i)==2 and ((7-j)==5)  :
                    obj_pairs = torch.cat([pan[:, 7 - i, :], pan[:, 7 - j, :]], 1)
                    obj_pairses_c[:, 0, :] = torch.cat([obj_pairs, ans], 1)
                    count -= 1

                # if its the last row, combine the context panels with the ans panel to create a triplet
                elif (7-i)==6 and ((7-j)==7) :
                    obj_pairs = torch.cat([pan[:, 7 - i, :], pan[:, 7 - j, :]], 1)
                    obj_pairses_r[:, 0, :] = torch.cat([obj_pairs, ans], 1)
                    count -= 1

                # else create triplets of other context panels and the ans panel
                else:
                    obj_pairs = torch.cat([pan[:,7-i,:],pan[:,7-j,:]],1)
                    obj_pairses[:,count,:] = torch.cat([obj_pairs, ans], 1)
                count+=1
        return obj_pairses,obj_pairses_c,obj_pairses_r

    # function g1:
    # used to get g-value of row and column ts
    def g_functin(self,context_pairs,panel_embedding_8,num_context_pairs,batch_size):
        context_pairs = torch.cat([context_pairs, panel_embedding_8.repeat(1, num_context_pairs, 1)], 2)
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 1024)
        context_g_out = self.g(context_pairs)
        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)
        context_g_out = context_g_out.sum(1)
        return context_g_out

    # function g2:
    # used to get g-values of non-row and non-column triplets
    def g_functin2(self,context_pairs,panel_embedding_8,num_context_pairs,batch_size):
        context_pairs = torch.cat([context_pairs, panel_embedding_8.repeat(1, num_context_pairs, 1)], 2)
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 1024)
        context_g_out = self.g2(context_pairs)
        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)
        context_g_out = context_g_out.sum(1)
        return context_g_out
    
    # main function
    def forward(self, x):
        # Note: the first 16 panels in x are shape panels, and the next 16 panels are line panels

        batch_size = x.shape[0]
        # Compute panel embeddings
        panel_embeddings = torch.zeros(batch_size, self.NUM_PANELS, 256).cuda()

        # get panel embedding of all line panels
        panel_embedding_8 = self.cnn_global(x[:, 16:, :, :])
        panel_embedding_8 = self.pre_g_fc2(panel_embedding_8.view(batch_size, 32 * 9 ** 2))
        panel_embedding_8 = self.pre_g_batch_norm2(panel_embedding_8)
        panel_embedding_8 = F.relu(panel_embedding_8)
        panel_embedding_8 = torch.unsqueeze(panel_embedding_8, 1)

        # get individual panel embeddings
        for panel_ind in range(self.NUM_PANELS):
            panel = x[:, panel_ind, :, :]       # shape panel
            panel2= x[:, panel_ind+16, :, :]    # line panel
            panel_embedding = self.comp_panel_embedding(panel, panel2)
            panel_embeddings[:, panel_ind, :] = panel_embedding
        context_embeddings = panel_embeddings[:, :int(self.NUM_PANELS/2), :] # (batch_size, 8, 256)
        answer_embeddings = panel_embeddings[:, int(self.NUM_PANELS/2):, :] # (batch_size, 8, 256)

        num_context_pairs = 56
        # Compute context pairs once to be used for each answer
        obj_pairses, obj_pairses_c, obj_pairses_r = self.panel_comp_obj_pairs(context_embeddings,batch_size)# (batch_size, 56, 256*3)

        # get g-values of all triplets than dont include answer panels
        '''context_pairs = torch.cat([context_pairs, panel_embedding_8.repeat(1, num_context_pairs, 1)], 2)
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 1024)
        context_g_out = self.g(context_pairs)
        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)'''
        context_g_out1= self.g_functin2(obj_pairses,panel_embedding_8,54,batch_size)
        context_g_outr = self.g_functin(obj_pairses_r, panel_embedding_8, 2, batch_size)
        context_g_outc = self.g_functin(obj_pairses_c, panel_embedding_8, 2, batch_size)
        # combine g-values of row, col, and other triplets into one
        context_g_out = context_g_out1 + context_g_outc + context_g_outr

        # placeholder for f-scores
        f_out = torch.zeros(batch_size, int(self.NUM_PANELS/2)).cuda()
        
        num_context_pairs = 28

        for answer_ind in range(8):
            answer_embedding = answer_embeddings[:, answer_ind, :] # (batch_size, 256)

            # combine answer panel with context panels to get row, col, and other triplets
            context_answer_pairs, context_answer_pairs_c,context_answer_pairs_r = self.ans_comp_obj_pairs(answer_embedding,context_embeddings,batch_size)# (batch_size, 28, 512)

            # get g-values for all answer panel triplets
            '''context_answer_pairs = torch.cat([context_answer_pairs, panel_embedding_8.repeat(1, 28, 1)], 2)
            context_answer_pairs = context_answer_pairs.view(batch_size * 28, 1024)
            context_answer_g_out = self.g(context_answer_pairs) # (8, 512)
            context_answer_g_out = context_answer_g_out.view(batch_size, 28, 512)
            context_answer_g_out = context_answer_g_out.sum(1)'''
            context_answer_g_out1 = self.g_functin2(context_answer_pairs, panel_embedding_8, 26, batch_size)
            context_answer_g_outr = self.g_functin(context_answer_pairs_r, panel_embedding_8, 1, batch_size)
            context_answer_g_outc = self.g_functin(context_answer_pairs_c, panel_embedding_8, 1, batch_size)

            # combine all answer panel g-values into one
            context_answer_g_out = context_answer_g_out1 + context_answer_g_outc  + context_answer_g_outr

            # combine non-answer panel g-values and answer panel g-values into one
            g_out = context_g_out + context_answer_g_out

            # get f-score
            f_out[:, answer_ind] = self.f(g_out).squeeze()
        
        return F.log_softmax(f_out, dim=1)