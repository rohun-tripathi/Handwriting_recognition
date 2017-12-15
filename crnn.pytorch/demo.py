import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


# Test input
img_path = './data/top.png'

# (1) For handwriting recognition + Text in wild - load GPU trained model
model_path = "./trained_models/netCRNN_43_500_0.667363636364.pth"

model = crnn.CRNN(32, 1, 37, 256)
loaded_model = torch.load(model_path, map_location=lambda storage, loc: storage)
loaded_state_dict = loaded_model['state']
new_net_state = model.state_dict()

for param_name in loaded_state_dict:
    new_name = param_name[7:]
    if new_name in new_net_state:
        new_net_state[new_name] = loaded_state_dict[param_name]

model.load_state_dict(new_net_state)

for p in model.parameters():
    p.requires_grad = False

# (2) For only text in wild - CPU model
# model_path = './trained_models/crnn.pth'
# model = crnn.CRNN(32, 1, 37, 256)
# if torch.cuda.is_available():
#     model = model.cuda()
# print('loading pretrained model from %s' % model_path)
# model.load_state_dict(torch.load(model_path))

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

non_maxed = preds

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

print('%-20s => %-20s' % (raw_pred, sim_pred))

print("Done")