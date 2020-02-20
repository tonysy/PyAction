import numpy as np
import torch
import math


class PositionEncoder3D(torch.nn.Module):
    """
    Implement various position embedding and conding method for 3d feature maps.
    "2d" means this method uses 2D conv
    "3d" means this method uses 3D conv and use frame number as temporal position code
    "trans" means use transformer temporal encoding method together with spatial pos embedding
    "temp" means learn spatial pos embedding first, learn temporal pos later
    """
    def __init__(self, channels, mode, multiplier=0.5, max_len=100):
        super(PositionEncoder3D, self).__init__()
        self.channels = channels
        assert(mode in ["concat2d", "elementwise2d", "concat3d", "elementwise3d",
                        "concat2dtrans", "elementwise2dtrans", "concat2dtemp", "elementwise2dtemp"])
        self.mode = mode
        if "2d" in mode:
            if "concat2d" in self.mode:
                self.l1 = torch.nn.Conv2d(channels+2, int(multiplier * channels), 1)
            elif "elementwise2d" in self.mode:
                self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
            self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)
        elif "3d" in mode:
            if "concat3d" in self.mode:
                self.l1 = torch.nn.Conv3d(channels+3, int(multiplier * channels), 1)
            elif "elementwise3d" in self.mode:
                self.l1 = torch.nn.Conv3d(3, int(multiplier * channels), 1)
            self.l2 = torch.nn.Conv3d(int(multiplier * channels), channels, 1)
        
        if "temp" in mode or "trans" in mode:
            pe = torch.zeros(max_len, channels) #maxlen: temporal lengths
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, channels, 2, dtype=torch.float) * -(math.log(10000.0) / channels)))
            pe[:, 0::2] = torch.sin(position.float() * div_term) # t c
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            self.pe = pe.cuda()
    
    def temposencode(self, x): #b c t lnum
        x_s = x.shape
        c_pe = self.pe[:x_s[2], :].transpose(1,0).unsqueeze(0).unsqueeze(-1)
        pos_embeds = x + c_pe
        return pos_embeds

    def forward(self, x):
        x_s = x.shape
        if len(x_s) <= 4 and "temp" in self.mode:
            return self.temposencode(x)

        pos_x = torch.arange(x_s[3]).float().unsqueeze(0).expand((x_s[4],-1)).unsqueeze(0)# 1, h, w
        pos_y = torch.arange(x_s[4]).float().unsqueeze(1).expand((-1,x_s[3])).unsqueeze(0)# 1, h, w
        pos_xy = torch.stack((pos_x,pos_y), dim=1)# 1, 2, h, w

        if "2d" in self.mode:
            pos = pos_xy
        elif "3d" in self.mode:
            pos_t = torch.arange(x_s[2]).float().unsqueeze(1) # t, 1
            pos_t = pos_t.expand((-1,x_s[3])).unsqueeze(2) # t, h, 1
            pos_t = pos_t.expand((-1,-1,x_s[4])).unsqueeze(0) # 1, t, h, w
            pos_xy = pos_xy.transpose(0,1).expand((-1,x_s[2],-1,-1)) # 2, t, h, w
            pos = torch.cat((pos_xy, pos_t), dim=0).unsqueeze(0) # 1, 3, t, h, w
        pos = pos.cuda()

        if "concat2d" in self.mode:
            pos = pos.unsqueeze(2)
            pos = pos.expand((x_s[0],-1,x_s[2],-1,-1)) # b, 2, t, h, w
            fea_pos = torch.cat((x,pos), dim=1)
            fea_pos = fea_pos.transpose(1,2) # b, t, 2, h, w
            fea_pos = fea_pos.reshape(-1,x_s[1]+2,x_s[3],x_s[4])
            pos_embeds = self.l2(torch.nn.functional.relu(self.l1(fea_pos)))
            pos_embeds = pos_embeds.reshape(x_s[0],x_s[2],x_s[1],x_s[3],x_s[4])
            pos_embeds = pos_embeds.transpose(1,2)
            if "trans" in self.mode:
                c_pe = self.pe[:x_s[2], :].transpose(1,0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                pos_embeds = pos_embeds + c_pe

        elif "elementwise2d" in self.mode:
            pos_embed = self.l2(torch.nn.functional.relu(self.l1(pos)))
            pos_embed = pos_embed.unsqueeze(2)
            pos_embeds = pos_embed.expand((x_s[0],-1,x_s[2],-1,-1))
            pos_embeds = x + pos_embeds
            if "trans" in self.mode:
                c_pe = self.pe[:x_s[2], :].transpose(1,0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                pos_embeds = pos_embeds + c_pe

        elif "concat3d" in self.mode:
            pos = pos.expand((x_s[0],-1,-1,-1,-1))
            fea_pos = torch.cat((x,pos), dim=1)
            pos_embeds = self.l2(torch.nn.functional.relu(self.l1(fea_pos)))

        elif "elementwise3d" in self.mode:
            pos_embed = self.l2(torch.nn.functional.relu(self.l1(pos)))
            pos_embeds = pos_embed.expand((x_s[0],-1,-1,-1,-1))
            pos_embeds = x + pos_embeds

        return pos_embeds


if __name__ == "__main__":
    fea = torch.ones(4, 64, 8, 26, 26).cuda()
    ml = list(["concat2dtrans", "elementwise2dtrans", "concat2dtemp", "elementwise2dtemp"])
    em1 = PositionEncoder3D(64, ml[0]).cuda()
    em2 = PositionEncoder3D(64, ml[1]).cuda()
    em3 = PositionEncoder3D(64, ml[2]).cuda()
    em4 = PositionEncoder3D(64, ml[3]).cuda()
    import ipdb; ipdb.set_trace()
