import torch

from stylegan import G_mapping, G_synthesis

'''
mapping = G_mapping().cuda()
load_model(self.mapping' path)
for idx in sample count:
    z = torch.randn((1,512))
    w_s=self.mapping(z) # size = 1x512
    w_s = w_s.unsqueeze(0) # 1x1x512
    w_s = torch.repeat(w_s, (1,18,1)) # 1x14x512
    img = G_synthesis(w_s, noise)
    score = classifier(img)
    data_pairs.append((w_s, score))
'''