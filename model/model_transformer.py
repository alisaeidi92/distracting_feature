import torch
import torch.nn.functional as F
from torch import nn

from .VisionTransformer import VisionTransformer

class MyTransformer(nn.Module):
    def __init__(self, args):
        super(MyTransformer, self).__init__()
        self.NUM_PANELS = 16
        self.type_loss = args.type_loss

        # print('________ running my transformer _________-')

        # to convert individual panels into embeddings
        self.transformer = VisionTransformer(
            image_size = 160,
            patch_size = 20,
            num_classes = 256,     # size of the panel embeddings
            dim = 1024,
            depth = 6,
            heads = 3,
            mlp_dim = 2048,
            dropout = 0.1,
            channels = 32
        )

        self.test_transformer = VisionTransformer(
            image_size = 160,
            patch_size = 20,     
            # num_classes = 2592,
            num_classes = 384,
            dim = 1024,       
            depth = 6, 
            heads = 3, 
            mlp_dim = 2048,
            dropout = 0.1,
            channels = 1  
        )

        self.transformer_global =  VisionTransformer(
            image_size = 160,
            patch_size = 20,     
            num_classes = 256,
            dim = 1024,       
            depth = 6, 
            heads = 3, 
            mlp_dim = 2048,
            dropout = 0.1,
            channels = 16   
        )


        # used to get embedding of individual panels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),     # (num_in_channels, num_out_channels, filter_size, stride_size)
            nn.BatchNorm2d(32),         # (num_features)
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

        # used to get the embedding of all context panels combined
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

        # used to get embedding of individual panels
        # self.pre_g_fc = nn.Linear(32 * 9 ** 2, 256)         # (in_features, out_features)
        self.pre_g_fc = nn.Linear(32 * 12, 256)
        self.pre_g_batch_norm = nn.BatchNorm1d(256)

        # used to get the embedding of all context panels combined
        self.pre_g_fc2 = nn.Linear(32 * 9 ** 2, 256)
        self.pre_g_batch_norm2 = nn.BatchNorm1d(256)

        # used in function g1 to get g-values of row and column ts
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

        # function f
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
        
        if self.type_loss:
            # used to calculate type loss
            self.meta_fc= nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, 9)
            )

    # given a panel, returns a panel embedding
    def comp_panel_embedding(self, panel):
        batch_size = panel.shape[0]
        panel = torch.unsqueeze(panel, 1)  # (batch_size, 160, 160) -> (batch_size, 1, 160, 160)
        # print('panel shape before cnn:', panel.shape)
        # panel_embedding = self.cnn(panel)  # (batch_size, 1, 160, 160) -> (batch_size, 32, 9, 9)
        panel_embedding = self.test_transformer(panel)
        # print('panel embedding shape after cnn:', panel_embedding.shape)
        panel_embedding = panel_embedding.view(batch_size, -1)
        # print(panel_embedding.shape)
        panel_embedding = self.pre_g_fc(panel_embedding)
        panel_embedding = self.pre_g_batch_norm(panel_embedding)
        panel_embedding = F.relu(panel_embedding)
        # print(panel_embedding.shape)
        return panel_embedding

    # outputs row, col, and other triplets that don't include an answer panel
    # objs: context panels
    def panel_comp_obj_pairs(self, objs, batch_size):
        # obj_pairses_r = torch.zeros(batch_size, 2, 256 * 3).cuda()
        # obj_pairses_c = torch.zeros(batch_size, 2, 256 * 3).cuda()
        # obj_pairses = torch.zeros(batch_size, 54, 256 * 3).cuda()
        obj_pairses_r = torch.zeros(batch_size, 2, 256 * 3)
        obj_pairses_c = torch.zeros(batch_size, 2, 256 * 3)
        obj_pairses = torch.zeros(batch_size, 54, 256 * 3)

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

                    # if the panels are the first or second column, combine them into one triplet
                    elif ((7-i)==0 and ((7-j)==3) and ((7-k)==6)) or((7-i)==1 and ((7-j)==4) and ((7-k)==7)):
                        obj_pairses_c[:, 7-i, :] = torch.cat(
                            [torch.cat([objs[:, 7 - i, :], objs[:, 7 - j, :]], 1), objs[:, 7 - k, :]], 1)
                        count -= 1
                    
                    # else combine the panels into pairs that have need an answer panel (ex: row 3 and col 3)
                    else:
                        obj_pairses[:,count,:] = torch.cat([torch.cat([objs[:, 7 - i, :], objs[:, 7 - j, :]],1), objs[:, 7-k, :]], 1)
                    #obj_pairs = torch.cat([torch.unsqueeze( objs[:,7-i,:],1),torch.unsqueeze( objs[:,7-j,:],1)],2)
                    #obj_pairses[:,count,:] = torch.cat([obj_pairs, torch.unsqueeze(objs[:, 7-k, :], 1)], 2)
                    count+=1
        return obj_pairses, obj_pairses_c, obj_pairses_r 


    # outputs row, col, and other triplets that include an answer panel
    # ans: answer panel embedding
    # pan: embeddings of context panels
    def ans_comp_obj_pairs(self, ans, pan, batch_size):
        # obj_pairses_r = torch.zeros(batch_size, 1, 256 * 3).cuda()
        # obj_pairses_c = torch.zeros(batch_size, 1, 256 * 3).cuda()
        # obj_pairses = torch.zeros(batch_size, 26, 256 * 3).cuda()
        obj_pairses_r = torch.zeros(batch_size, 1, 256 * 3)
        obj_pairses_c = torch.zeros(batch_size, 1, 256 * 3)
        obj_pairses = torch.zeros(batch_size, 26, 256 * 3)

        count=0
        for i in range(8):
            for j in range(i):
                # if its the last column, combine the context panels with the ans panel to make a triplet
                if (7-i)==2 and ((7-j)==5)  :
                    obj_pairs = torch.cat([pan[:, 7 - i, :], pan[:, 7 - j, :]], 1)
                    obj_pairses_c[:, 0, :] = torch.cat([obj_pairs, ans], 1)
                    count -= 1

                # if its the last row, combine the context panels with the ans panel to make a triplet
                elif (7-i)==6 and ((7-j)==7) :
                    obj_pairs = torch.cat([pan[:, 7 - i, :], pan[:, 7 - j, :]], 1)
                    obj_pairses_r[:, 0, :] = torch.cat([obj_pairs, ans], 1)
                    count -= 1

                # else create triplets of other context panels and the ans panel
                else:
                    obj_pairs = torch.cat([pan[:, 7 - i, :], pan[:, 7 - j, :]], 1)
                    obj_pairses[:,count,:] = torch.cat([obj_pairs, ans], 1)
                count+=1
        return obj_pairses, obj_pairses_c, obj_pairses_r


    # function g1:
    # used to get g-value of row and column triplets
    def g_functin(self, context_pairs, panel_embedding_8, num_context_pairs, batch_size):
        context_pairs = torch.cat([context_pairs, panel_embedding_8.repeat(1, num_context_pairs, 1)], 2)
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 1024)
        context_g_out = self.g(context_pairs)
        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)
        context_g_out = context_g_out.sum(1)
        return context_g_out


    # function g2:
    # used to get g-values of non-row and non-col triplets
    def g_functin2(self, context_pairs, panel_embedding_8, num_context_pairs, batch_size):
        context_pairs = torch.cat([context_pairs, panel_embedding_8.repeat(1, num_context_pairs, 1)], 2)
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 1024)
        context_g_out = self.g2(context_pairs)
        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)
        context_g_out = context_g_out.sum(1)
        return context_g_out

    
    # main function
    def forward(self, x):
        """
        general steps:
        1. get embeddings of all context panels (8) and all answer panels(8)
        2. get all combinations of triplets that don't require answer panels
        3. get g-values of all triplets from step 2
        4: combine all g-values from step 3 into one
        5. get all combinations of triplets that require answer panels
        6. get g-values of all triplets from step 5
        7: combine all g-values from step 6 into one
        8: combine both g-values from steps 4 and 7 into one
        9: get f-score using g-value from step 8
        10: output softmax of f-score, and type loss
        """

        # x.shape is [32, 16, 160, 160]

        batch_size = x.shape[0]

        # placeholder for panel embeddings
        # panel_embeddings = torch.zeros(batch_size, self.NUM_PANELS, 256).cuda()
        panel_embeddings = torch.zeros(batch_size, self.NUM_PANELS, 256)

        # print('x.shape:', x.shape)

        # an embedding of all panels (the yellow embedding in the diagram)
        panel_embedding_8 = self.cnn_global(x[:, :, :, :])
        panel_embedding_8 = self.pre_g_fc2(panel_embedding_8.view(batch_size, -1))
        panel_embedding_8 = self.pre_g_batch_norm2(panel_embedding_8)
        panel_embedding_8 = F.relu(panel_embedding_8)
        panel_embedding_8 = torch.unsqueeze(panel_embedding_8, 1)
        # print('panel_embedding_8.shape', panel_embedding_8.shape)

        # panel_embedding_8 = self.transformer_global(x)
        # shape = (panel_embedding_8.shape[0], 1, panel_embedding_8.shape[1])     # (32, 1, 256)
        # panel_embedding_8 = torch.reshape(panel_embedding_8, shape)

        # get panel embeddings for all panels (8 context panels, 8 answer panels)
        for panel_ind in range(self.NUM_PANELS):
            panel = x[:, panel_ind, :, :]
            # print('--- panel.shape', panel.shape)
            # panel = panel[None, :, :, :]
            # print('--- panel.shape', panel.shape)
            # panel_embedding = self.transformer(panel)
            # print('panel embedding shape:', panel_embedding.shape)
            panel_embedding = self.comp_panel_embedding(panel)
            panel_embeddings[:, panel_ind, :] = panel_embedding

        context_embeddings = panel_embeddings[:, :int(self.NUM_PANELS/2), :] # (batch_size, 8, 256)
        answer_embeddings = panel_embeddings[:, int(self.NUM_PANELS/2):, :] # (batch_size, 8, 256)


        num_context_pairs = 56
        # Compute context pairs once to be used for each answer
        obj_pairses, obj_pairses_c, obj_pairses_r = self.panel_comp_obj_pairs(context_embeddings, batch_size) # (batch_size, 56, 256*3)

        # get g-values of all triplets that dont include answer panels
        '''context_pairs = torch.cat([context_pairs, panel_embedding_8.repeat(1, num_context_pairs, 1)], 2)
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 1024)
        context_g_out = self.g(context_pairs)
        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)'''
        context_g_out1 = self.g_functin2(obj_pairses, panel_embedding_8, 54, batch_size)
        context_g_outr = self.g_functin(obj_pairses_r, panel_embedding_8, 2, batch_size)
        context_g_outc = self.g_functin(obj_pairses_c, panel_embedding_8, 2, batch_size)
        # combine g-values of row-triplets, col-triplets, and all other triplets into one
        context_g_out = context_g_out1 + context_g_outc + context_g_outr

        # placeholder for f-scores
        # f_out = torch.zeros(batch_size, int(self.NUM_PANELS/2)).cuda()
        f_out = torch.zeros(batch_size, int(self.NUM_PANELS/2))

        # placeholder for type loss
        if self.type_loss:
            # f_meta=torch.zeros(batch_size, 512).cuda()
            f_meta=torch.zeros(batch_size, 512)

        for answer_ind in range(8):
            # get individual answer panel embedding
            answer_embedding = answer_embeddings[:, answer_ind, :] # (batch_size, 256)

            # get row, column, and other triplets using current answer panel and all context panels
            context_answer_pairs, context_answer_pairs_c, context_answer_pairs_r = self.ans_comp_obj_pairs(answer_embedding,context_embeddings, batch_size)# (batch_size, 28, 512)

            # apply g-values for all answer panel triplets
            '''context_answer_pairs = torch.cat([context_answer_pairs, panel_embedding_8.repeat(1, 28, 1)], 2)
            context_answer_pairs = context_answer_pairs.view(batch_size * 28, 1024)
            context_answer_g_out = self.g(context_answer_pairs) # (8, 512)
            context_answer_g_out = context_answer_g_out.view(batch_size, 28, 512)
            context_answer_g_out = context_answer_g_out.sum(1)'''
            context_answer_g_out1 = self.g_functin2(context_answer_pairs, panel_embedding_8, 26, batch_size)
            context_answer_g_outr = self.g_functin(context_answer_pairs_r, panel_embedding_8, 1, batch_size)
            context_answer_g_outc = self.g_functin(context_answer_pairs_c, panel_embedding_8, 1, batch_size)

            # combine g-values of row, column, and other triplets into one
            context_answer_g_out = context_answer_g_out1 + context_answer_g_outc  + context_answer_g_outr

            # combine context-panel-only and answer-panel g-values into one
            g_out = context_g_out + context_answer_g_out

            if self.type_loss:
                f_meta+=g_out

            # get f-score
            f_out[:, answer_ind] = self.f(g_out).squeeze()
            # print('---****** f_out.shape:',f_out.shape)
        
        # print('model output shape:', f_out.shape)
        # output the softmax of the f-scores (and type-loss if asked)
        if self.type_loss:
            return F.log_softmax(f_out, dim=1),F.sigmoid(self.meta_fc(f_meta))
        else:
            return F.log_softmax(f_out, dim=1)
