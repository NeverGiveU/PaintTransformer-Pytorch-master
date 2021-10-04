import torch
import torch.nn as nn 
import math 
import copy 


class Painter(nn.Module):
    def __init__(self, n_param_stroke, n_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3):
        super(Painter, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_param_stroke = n_param_stroke
        ## backbone for feature extraction
        self.enc_target = nn.Sequential(
            # block 1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # block 2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # block 3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.enc_canvas = copy.deepcopy(self.enc_target)
        self.conv = nn.Conv2d(128*2, hidden_dim, 1) # 1x1 conv
        
        ## transformer for stroke pamareters prediction
        '''
        nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, 
                       num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, 
                       activation='relu', custom_encoder=None, custom_decoder=None)
        '''
        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers, batch_first=True)
        ## two heads
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_param_stroke)
        )
        self.linear_score = nn.Linear(hidden_dim, 1)

        ## 
        self.query_pos_embedding = nn.Parameter(torch.FloatTensor(n_strokes, hidden_dim), requires_grad=True) # input of the decoder
        self.row_embedding = nn.Parameter(torch.FloatTensor(n_heads, hidden_dim//2), requires_grad=True)
        self.col_embedding = nn.Parameter(torch.FloatTensor(n_heads, hidden_dim//2), requires_grad=True)
        self.reset()

    def reset(self):
        nn.init.uniform_(self.query_pos_embedding, 0., 1.)
        nn.init.uniform_(self.row_embedding, 0., 1.)
        nn.init.uniform_(self.col_embedding, 0., 1.)

    def forward(self, target, canvas):
        target_feat = self.enc_target(target)
        canvas_feat = self.enc_canvas(canvas)

        feat = torch.cat((target_feat, canvas_feat), dim=1)
        feat = self.conv(feat)                                   # (b,c,h,w)
        
        b, c, h, w = feat.size()
        pos_embedding = torch.cat([
            self.col_embedding[:w].unsqueeze(0).repeat(h, 1, 1), # (h,w,c//2)
            self.row_embedding[:h].unsqueeze(1).repeat(1, w, 1), # (h,w,c//2)
        ], dim=-1).view(-1, self.hidden_dim).unsqueeze(1)        # (hxw,1,self.c)
        
        src = (pos_embedding + feat.view(b, c, -1).permute(2, 0, 1)).permute(1, 0, 2)  # (b,hxw,c)
        tgt = self.query_pos_embedding.unsqueeze(1).repeat(1, b, 1).permute(1, 0, 2)   # (b,n_strokes,c)
        hidden_state = self.transformer(src, tgt)                                      # (b,n_strokes,c)
        # print(hidden_state)
        
        param = self.linear_param(hidden_state)                                        # (b,n_strokes,n_param_stroke)
        score = self.linear_score(hidden_state)
        return param, score 


## testing
# painter = Painter(5, 8, 256)
# for b in [1, 2, 2, 3, 4, 8, 9, 16, 17]:
#     t = torch.randn(b, 3, 32, 32)
#     c = torch.randn(b, 3, 32, 32)
#     p, s = painter(t, c)
#     print(p.size(), s.size())